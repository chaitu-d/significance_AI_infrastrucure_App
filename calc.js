/* Pure calculation engine for the AI Infrastructure Suite. No DOM access. */
(function (root, factory) {
    const api = factory();
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    } else {
        root.AICalc = api;
    }
})(typeof self !== 'undefined' ? self : globalThis, function () {
    'use strict';

    // Approximate public datasheet values. Edit to match your own quote or datasheet.
    const GPUS = {
        'NVIDIA H100 SXM (80 GB)': { tflopsBF16: 989, vramGB: 80, tdpW: 700 },
        'NVIDIA H200 (141 GB)': { tflopsBF16: 989, vramGB: 141, tdpW: 700 },
        'NVIDIA B200 (192 GB)': { tflopsBF16: 2250, vramGB: 192, tdpW: 1000 },
        'NVIDIA A100 SXM (80 GB)': { tflopsBF16: 312, vramGB: 80, tdpW: 400 },
        'NVIDIA L40S (48 GB)': { tflopsBF16: 362, vramGB: 48, tdpW: 350 },
    };

    const BYTES_PER_PARAM = { 'FP32': 4, 'FP16 / BF16': 2, 'INT8': 1, 'INT4': 0.5 };
    const TRAIN_BYTES_PER_PARAM = 16; // mixed precision + Adam: weights, grads, fp32 master, two optimizer states
    const CHECKPOINT_BYTES_PER_PARAM = 14; // weights + fp32 master + optimizer states
    const INFERENCE_OVERHEAD = 0.2; // runtime and activation buffers
    const BYTES_PER_TOKEN = 4; // int32 token IDs
    const HOURS_PER_YEAR = 8760;

    const SECURITY_CONTROLS = [
        { id: 'zeroTrust', name: 'Zero-trust access', weight: 15, tip: 'Verify every user, device, and workload on each request.' },
        { id: 'encryption', name: 'Encryption at rest and in transit', weight: 15, tip: 'TLS for data in motion; strong encryption for datasets and models at rest.' },
        { id: 'iam', name: 'Identity and access management', weight: 10, tip: 'Role- or attribute-based access control with multi-factor authentication.' },
        { id: 'segmentation', name: 'Network segmentation', weight: 10, tip: 'Isolated zones and micro-segmentation to limit lateral movement.' },
        { id: 'modelIntegrity', name: 'Model integrity validation', weight: 10, tip: 'Hashes or signatures confirm models have not been tampered with.' },
        { id: 'adversarial', name: 'Adversarial-input defenses', weight: 10, tip: 'Input validation and robustness testing against manipulated inputs.' },
        { id: 'poisoning', name: 'Data poisoning prevention', weight: 10, tip: 'Dataset provenance checks and validated, secure data pipelines.' },
        { id: 'modelAccess', name: 'Protection against model theft', weight: 10, tip: 'Encrypted model weights and audited access to model artifacts.' },
        { id: 'compliance', name: 'Privacy compliance (GDPR / HIPAA)', weight: 5, tip: 'Applicable regulations are mapped to data flows and policies.' },
        { id: 'monitoring', name: 'Continuous monitoring and logging', weight: 5, tip: 'Centralized logs and alerting (SIEM) for traceability.' },
    ];
    const LEVELS = { none: 0, partial: 0.5, full: 1 };

    const isPos = (x) => Number.isFinite(x) && x > 0;
    const isNonNeg = (x) => Number.isFinite(x) && x >= 0;

    // Training FLOPs ~= 6 * parameters * tokens (standard transformer estimate).
    function trainingEstimate({ paramsB, tokensB, gpuCount, gpu, mfu, pricePerGpuHour }) {
        const spec = GPUS[gpu];
        if (!spec || !isPos(paramsB) || !isPos(tokensB) || !isPos(gpuCount) || !isPos(mfu) || mfu > 1) return null;
        const flops = 6 * paramsB * 1e9 * tokensB * 1e9;
        const effectiveFlops = gpuCount * spec.tflopsBF16 * 1e12 * mfu;
        const seconds = flops / effectiveFlops;
        const gpuHours = (gpuCount * seconds) / 3600;
        return {
            flops,
            effectiveFlops,
            days: seconds / 86400,
            gpuHours,
            costUSD: isNonNeg(pricePerGpuHour) ? gpuHours * pricePerGpuHour : null,
            tokensPerSec: effectiveFlops / (6 * paramsB * 1e9),
        };
    }

    function inferenceMemory({ paramsB, bytesPerParam, kvCacheGB, vramGB }) {
        if (!isPos(paramsB) || !isPos(bytesPerParam) || !isNonNeg(kvCacheGB) || !isPos(vramGB)) return null;
        const weightsGB = paramsB * bytesPerParam;
        const totalGB = weightsGB * (1 + INFERENCE_OVERHEAD) + kvCacheGB;
        return { weightsGB, totalGB, gpusNeeded: Math.ceil(totalGB / vramGB) };
    }

    function trainingMemory({ paramsB, activationOverhead, vramGB }) {
        if (!isPos(paramsB) || !isNonNeg(activationOverhead) || !isPos(vramGB)) return null;
        const stateGB = paramsB * TRAIN_BYTES_PER_PARAM;
        const totalGB = stateGB * (1 + activationOverhead);
        return { stateGB, totalGB, gpusNeededSharded: Math.ceil(totalGB / vramGB) };
    }

    function coolingAdvice(rackKw) {
        if (rackKw <= 15) return 'Standard air cooling is usually sufficient.';
        if (rackKw <= 30) return 'Enhanced air cooling or rear-door heat exchangers are recommended.';
        if (rackKw <= 50) return 'Direct-to-chip liquid cooling is recommended.';
        return 'Immersion or full liquid cooling is recommended; confirm facility capacity.';
    }

    function powerEstimate({ gpuCount, gpu, serverOverhead, pue, utilization, kwhPrice, co2PerKwh, rackKw }) {
        const spec = GPUS[gpu];
        if (!spec || !isPos(gpuCount) || !isNonNeg(serverOverhead) || !(pue >= 1) || !(utilization >= 0 && utilization <= 1)
            || !isNonNeg(kwhPrice) || !isNonNeg(co2PerKwh) || !isPos(rackKw)) return null;
        const gpuKw = (gpuCount * spec.tdpW) / 1000;
        const itKw = gpuKw * (1 + serverOverhead);
        const facilityKw = itKw * pue;
        const annualKwh = facilityKw * HOURS_PER_YEAR * utilization;
        return {
            gpuKw,
            itKw,
            coolingKw: facilityKw - itKw,
            facilityKw,
            racks: Math.ceil(itKw / rackKw),
            annualKwh,
            annualCostUSD: annualKwh * kwhPrice,
            annualCo2Tons: (annualKwh * co2PerKwh) / 1000,
            cooling: coolingAdvice(rackKw),
        };
    }

    // Leaf-spine fabric: each leaf gives `uplinks` ports to spines and `downlinks` ports to endpoints.
    function fabricEstimate({ endpoints, radix, oversub, linkGbps }) {
        if (!isPos(endpoints) || !isPos(radix) || radix < 4 || !isPos(oversub) || !isPos(linkGbps)) return null;
        const uplinks = Math.floor(radix / (1 + oversub));
        if (uplinks < 1) return null;
        const downlinks = radix - uplinks;
        const leaves = Math.ceil(endpoints / downlinks);
        const spines = uplinks;
        const tiers = leaves <= radix ? 2 : 3;
        const actualOversub = downlinks / uplinks;
        return {
            uplinks,
            downlinks,
            leaves,
            spines,
            switches: tiers === 2 ? leaves + spines : null,
            tiers,
            actualOversub,
            perEndpointGbps: linkGbps / actualOversub,
            fabricTbps: (endpoints * linkGbps) / 1000,
        };
    }

    function storageEstimate({ datasetTB, replication, paramsB, checkpointsKept, checkpointWindowSec, tokensPerSec }) {
        if (!isNonNeg(datasetTB) || !isPos(replication) || !isPos(paramsB) || !isNonNeg(checkpointsKept)
            || !isPos(checkpointWindowSec)) return null;
        const checkpointGB = paramsB * CHECKPOINT_BYTES_PER_PARAM;
        const checkpointsTB = (checkpointGB * checkpointsKept) / 1000;
        const datasetTotalTB = datasetTB * replication;
        return {
            checkpointGB,
            checkpointsTB,
            datasetTotalTB,
            totalTB: datasetTotalTB + checkpointsTB,
            writeGBps: checkpointGB / checkpointWindowSec,
            ingestMBps: isPos(tokensPerSec) ? (tokensPerSec * BYTES_PER_TOKEN) / 1e6 : null,
        };
    }

    // levels: { controlId: 'none' | 'partial' | 'full' }
    function securityScore(levels) {
        let score = 0;
        const gaps = [];
        SECURITY_CONTROLS.forEach((c) => {
            const level = LEVELS[levels[c.id]] ?? 0;
            score += c.weight * level;
            if (level < 1) gaps.push({ name: c.name, missing: c.weight * (1 - level) });
        });
        gaps.sort((a, b) => b.missing - a.missing);
        const rating = score >= 85 ? 'Strong' : score >= 60 ? 'Moderate' : 'Needs work';
        return { score, rating, gaps };
    }

    return {
        GPUS,
        BYTES_PER_PARAM,
        SECURITY_CONTROLS,
        trainingEstimate,
        inferenceMemory,
        trainingMemory,
        coolingAdvice,
        powerEstimate,
        fabricEstimate,
        storageEstimate,
        securityScore,
    };
});
