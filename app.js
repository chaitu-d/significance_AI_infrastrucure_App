(function () {
    'use strict';

    const C = window.AICalc;
    const DEFAULT_GPU = 'NVIDIA H100 SXM (80 GB)';
    const DEFAULT_SECURITY = { zeroTrust: 'full', encryption: 'full', iam: 'partial', segmentation: 'partial' };
    const LEVEL_LABELS = { none: 'Not in place', partial: 'Partial', full: 'Fully in place' };

    const $ = (id) => document.getElementById(id);
    const num = (id) => parseFloat($(id).value);
    const txt = (id) => $(id).value;
    const set = (id, s) => { $(id).textContent = s; };

    const fmt = (x, d = 0) => (Number.isFinite(x)
        ? x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })
        : '—');
    const usd = (x) => (Number.isFinite(x) ? '$' + fmt(x) : '—');
    const gb = (x) => (Number.isFinite(x) ? fmt(x, 1) + ' GB' : '—');
    const dur = (days) => (!Number.isFinite(days) ? '—'
        : days < 1 ? fmt(days * 24, 1) + ' hours' : fmt(days, 1) + ' days');

    function buildControls() {
        const gpuSel = $('gpuModel');
        Object.keys(C.GPUS).forEach((name) => gpuSel.add(new Option(name, name)));
        gpuSel.value = DEFAULT_GPU;

        const list = $('secList');
        C.SECURITY_CONTROLS.forEach((c) => {
            const row = document.createElement('div');
            row.className = 'control';

            const info = document.createElement('div');
            const name = document.createElement('strong');
            name.textContent = c.name;
            const weight = document.createElement('span');
            weight.className = 'weight';
            weight.textContent = `${c.weight} pts`;
            name.appendChild(weight);
            const tip = document.createElement('p');
            tip.textContent = c.tip;
            info.append(name, tip);

            const sel = document.createElement('select');
            sel.id = 'sec-' + c.id;
            Object.entries(LEVEL_LABELS).forEach(([value, label]) => sel.add(new Option(label, value)));
            sel.value = DEFAULT_SECURITY[c.id] || 'none';

            row.append(info, sel);
            list.appendChild(row);
        });
    }

    function showTab(name) {
        document.querySelectorAll('.panel').forEach((p) => { p.hidden = p.id !== 'panel-' + name; });
        document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
        history.replaceState(null, '', '#' + name);
        window.scrollTo(0, 0);
    }

    function renderSecurity(sec) {
        set('outSecScore', sec ? fmt(sec.score) + ' / 100' : '—');
        const badge = $('outSecRating');
        badge.textContent = sec ? sec.rating : '—';
        const cls = sec ? { Strong: 'strong', Moderate: 'moderate', 'Needs work': 'weak' }[sec.rating] : '';
        badge.className = 'badge' + (cls ? ' ' + cls : '');
        $('secBar').style.width = sec ? sec.score + '%' : '0';

        const gaps = $('secGaps');
        gaps.replaceChildren();
        const items = sec ? sec.gaps.slice(0, 5) : [];
        if (!items.length) {
            const li = document.createElement('li');
            li.textContent = sec ? 'All controls are fully in place.' : 'Enter valid settings.';
            gaps.appendChild(li);
        }
        items.forEach((g) => {
            const li = document.createElement('li');
            li.textContent = `${g.name}: +${fmt(g.missing)} points available`;
            gaps.appendChild(li);
        });
    }

    function recalc() {
        const gpu = txt('gpuModel');
        const spec = C.GPUS[gpu];
        const gpuCount = Math.floor(num('gpuCount'));
        const pue = num('pue');
        const kwhPrice = num('kwhPrice');
        set('clusterSpec', `${spec.vramGB} GB VRAM · ${fmt(spec.tflopsBF16)} TFLOPS BF16 · ${spec.tdpW} W TDP per GPU`);

        // Compute
        const train = C.trainingEstimate({
            paramsB: num('compParams'),
            tokensB: num('compTokens'),
            gpuCount,
            gpu,
            mfu: num('compMfu') / 100,
            pricePerGpuHour: num('compPrice'),
        });
        set('outDays', train ? dur(train.days) : '—');
        set('outGpuHours', train ? fmt(train.gpuHours) : '—');
        set('outTrainCost', train ? usd(train.costUSD) : '—');
        set('outFlops', train ? train.flops.toExponential(2) : '—');
        set('outTokPerSec', train ? fmt(train.tokensPerSec) : '—');

        // Memory
        const modelB = num('memParams');
        const inf = C.inferenceMemory({
            paramsB: modelB,
            bytesPerParam: C.BYTES_PER_PARAM[txt('memPrecision')],
            kvCacheGB: num('memKv'),
            vramGB: spec.vramGB,
        });
        set('outInfWeights', inf ? gb(inf.weightsGB) : '—');
        set('outInfTotal', inf ? gb(inf.totalGB) : '—');
        set('outInfGpus', inf ? fmt(inf.gpusNeeded) : '—');
        const trn = C.trainingMemory({
            paramsB: modelB,
            activationOverhead: num('memAct') / 100,
            vramGB: spec.vramGB,
        });
        set('outTrainState', trn ? gb(trn.stateGB) : '—');
        set('outTrainTotal', trn ? gb(trn.totalGB) : '—');
        set('outTrainGpus', trn ? fmt(trn.gpusNeededSharded) : '—');

        // Power and cooling
        const pow = C.powerEstimate({
            gpuCount,
            gpu,
            serverOverhead: num('powOverhead') / 100,
            pue,
            utilization: num('powUtil') / 100,
            kwhPrice,
            co2PerKwh: num('powCo2'),
            rackKw: num('powRack'),
        });
        set('outGpuKw', pow ? fmt(pow.gpuKw) + ' kW' : '—');
        set('outItKw', pow ? fmt(pow.itKw) + ' kW' : '—');
        set('outCoolKw', pow ? fmt(pow.coolingKw) + ' kW' : '—');
        set('outFacKw', pow ? fmt(pow.facilityKw) + ' kW' : '—');
        set('outRacks', pow ? fmt(pow.racks) : '—');
        set('outKwh', pow ? fmt(pow.annualKwh / 1e6, 2) + ' GWh' : '—');
        set('outEnergyCost', pow ? usd(pow.annualCostUSD) : '—');
        set('outCo2', pow ? fmt(pow.annualCo2Tons) + ' t' : '—');
        set('outCooling', pow ? pow.cooling : '—');
        const itShare = pow ? (pow.itKw / pow.facilityKw) * 100 : 0;
        $('powBarIt').style.width = itShare + '%';
        $('powBarCool').style.width = (100 - itShare) + '%';

        // Network
        const endpoints = gpuCount * num('netNics');
        const fab = C.fabricEstimate({
            endpoints,
            radix: num('netRadix'),
            oversub: num('netOversub'),
            linkGbps: num('netSpeed'),
        });
        set('outEndpoints', Number.isFinite(endpoints) ? fmt(endpoints) : '—');
        set('outTiers', fab ? (fab.tiers === 2 ? '2-tier leaf-spine' : '3-tier (needs redesign)') : '—');
        set('outLeaves', fab ? fmt(fab.leaves) : '—');
        set('outSpines', fab ? fmt(fab.spines) : '—');
        set('outSwitches', fab ? (fab.switches == null ? 'Not estimated' : fmt(fab.switches)) : '—');
        set('outUplinks', fab ? `${fab.downlinks} / ${fab.uplinks}` : '—');
        set('outFabricTbps', fab ? fmt(fab.fabricTbps, 1) + ' Tbps' : '—');
        set('outPerEp', fab ? fmt(fab.perEndpointGbps, 0) + ' Gbps' : '—');

        // Storage
        const st = C.storageEstimate({
            datasetTB: num('stDataset'),
            replication: num('stRep'),
            paramsB: num('stModel'),
            checkpointsKept: num('stKept'),
            checkpointWindowSec: num('stWindow'),
            tokensPerSec: train ? train.tokensPerSec : null,
        });
        set('outCkpt', st ? gb(st.checkpointGB) : '—');
        set('outCkptTotal', st ? fmt(st.checkpointsTB, 2) + ' TB' : '—');
        set('outDataTotal', st ? fmt(st.datasetTotalTB, 1) + ' TB' : '—');
        set('outStTotal', st ? fmt(st.totalTB, 1) + ' TB' : '—');
        set('outCkptBw', st ? fmt(st.writeGBps, 2) + ' GB/s' : '—');
        set('outIngest', st && st.ingestMBps != null ? fmt(st.ingestMBps, 1) + ' MB/s' : '—');

        // Security
        const levels = {};
        C.SECURITY_CONTROLS.forEach((c) => { levels[c.id] = txt('sec-' + c.id); });
        const sec = C.securityScore(levels);
        renderSecurity(sec);

        // Dashboard
        set('dbGpus', fmt(gpuCount));
        set('dbGpuSub', `${gpu} · ${fmt(gpuCount * spec.tflopsBF16 / 1000, 1)} PFLOPS BF16 peak`);
        set('dbTrain', train ? dur(train.days) : '—');
        set('dbTrainSub', train ? `${fmt(modelB, 1)}B params on ${fmt(num('compTokens'))}B tokens` : '—');
        set('dbIt', pow ? fmt(pow.itKw) + ' kW' : '—');
        set('dbFac', pow ? fmt(pow.facilityKw / 1000, 2) + ' MW' : '—');
        set('dbFacSub', pow ? `PUE ${fmt(pue, 2)} · ${fmt(pow.racks)} racks` : '—');
        set('dbCost', pow ? usd(pow.annualCostUSD) : '—');
        set('dbKwh', pow ? fmt(pow.annualKwh / 1e6, 2) + ' GWh per year' : '—');
        set('dbCo2', pow ? fmt(pow.annualCo2Tons) + ' t' : '—');
        set('dbFabric', fab && fab.switches != null ? fmt(fab.switches) + ' switches' : '—');
        set('dbFabricSub', fab ? `${fab.tiers}-tier · ${fmt(fab.fabricTbps, 1)} Tbps aggregate` : '—');
        set('dbInf', inf ? fmt(inf.gpusNeeded) : '—');
        set('dbInfSub', inf ? `GPUs for ${fmt(modelB, 1)}B params at ${txt('memPrecision')}` : '—');
        set('dbSec', sec ? fmt(sec.score) : '—');
        set('dbSecSub', sec ? `${sec.rating} security posture` : '—');

        renderNotes({ train, pow, fab, st, sec, pue, rackKw: num('powRack') });
    }

    function renderNotes({ train, pow, fab, st, sec, pue, rackKw }) {
        const notes = [];
        if (sec && sec.rating !== 'Strong' && sec.gaps[0]) {
            notes.push(`Security score is ${fmt(sec.score)} out of 100. Largest gap: ${sec.gaps[0].name}.`);
        }
        if (pow && pow.facilityKw > 1000) {
            notes.push(`Facility draw is ${fmt(pow.facilityKw / 1000, 2)} MW. Confirm utility capacity, backup power, and grid connection timelines.`);
        }
        if (pow && rackKw > 30) {
            notes.push(`Rack density of ${fmt(rackKw)} kW: ${pow.cooling}`);
        }
        if (pue > 1.5) {
            notes.push(`PUE of ${fmt(pue, 2)} wastes a large share of energy. Modern AI facilities typically target a PUE near 1.1 to 1.3.`);
        }
        if (fab && fab.tiers === 3) {
            notes.push('The fabric needs three tiers. Use higher-radix switches or a lower oversubscription ratio to keep it two-tier.');
        }
        if (train && train.days > 30) {
            notes.push(`The training run takes ${fmt(train.days, 0)} days. Adding GPUs or raising utilization shortens the schedule.`);
        }
        if (st && st.writeGBps >= 5) {
            notes.push(`Checkpoint writes need ${fmt(st.writeGBps, 1)} GB/s. Plan for a parallel file system or NVMe-oF storage, or lengthen the write window.`);
        }
        if (!notes.length) notes.push('No critical issues detected with the current inputs.');

        const list = $('dbNotes');
        list.replaceChildren();
        notes.forEach((text) => {
            const li = document.createElement('li');
            li.textContent = text;
            list.appendChild(li);
        });
    }

    function init() {
        buildControls();

        document.addEventListener('input', recalc);
        document.addEventListener('change', recalc);
        document.addEventListener('click', (e) => {
            const target = e.target.closest('[data-tab], [data-go]');
            if (target) showTab(target.dataset.tab || target.dataset.go);
        });

        const start = location.hash.slice(1);
        showTab(start && document.getElementById('panel-' + start) ? start : 'dashboard');
        recalc();
    }

    document.addEventListener('DOMContentLoaded', init);
})();
