/* Verified-HAB mode is independent of the archived chlorophyll dashboard. */
let habForecast = null;
let habRequest = 0;
const habColors = {LOW:'#a8d8b9', MODERATE:'#f6cf71', HIGH:'#ed9e93', UNAVAILABLE:'#cdd6d9'};
const habElement = id => document.getElementById(id);

function habClear() {
    habForecast = null;
    habElement('hab-results').hidden = true;
    habElement('hab-rows').replaceChildren();
    habElement('hab-map').replaceChildren();
    habElement('hab-zone-detail').textContent = '';
}

function habDetail(zone) {
    const q = zone.data_quality;
    habElement('hab-zone-detail').textContent = `${zone.name}: ${q.observation_status}. Mean chlorophyll input coverage ${q.chlorophyll_mean_input_coverage_percent.toFixed(1)}%. Insufficient variables: ${q.insufficient_input_features.join(', ') || 'none'}. Coverage gates are prototype safeguards.`;
}

function habRender(day) {
    if (!habForecast) return;
    const row = habForecast.daily_predictions[day-1];
    habElement('hab-context').textContent = `Day +${day} · ${row.date}. Inputs end ${habForecast.input_dates.at(-1)} (${habForecast.freshness.state}). Experimental calibration; delivery-based lead time is unverified.`;
    const body = habElement('hab-rows');
    const svg = habElement('hab-map');
    body.replaceChildren(); svg.replaceChildren();
    const zones = habForecast.zones;
    const west = Math.min(...zones.map(z=>z.bounds[0])), south = Math.min(...zones.map(z=>z.bounds[1]));
    const east = Math.max(...zones.map(z=>z.bounds[2])), north = Math.max(...zones.map(z=>z.bounds[3]));
    const x = lon=>45+(lon-west)/(east-west)*550, y = lat=>365-(lat-south)/(north-south)*320;
    const make = (name, attributes, text) => {
        const node = document.createElementNS('http://www.w3.org/2000/svg', name);
        Object.entries(attributes).forEach(([key,value])=>node.setAttribute(key,String(value)));
        if(text) node.textContent=text;
        return node;
    };
    zones.forEach(zone=>{
        const prediction = row.zones.find(z=>z.zone_id===zone.id);
        const tr = document.createElement('tr');
        const name = document.createElement('td'), button=document.createElement('button');
        button.type='button'; button.textContent=zone.name; button.addEventListener('click',()=>habDetail(zone)); name.append(button); tr.append(name);
        [prediction.probability === null ? 'Unavailable' : `${(prediction.probability*100).toFixed(1)}% (experimental)`, prediction.risk_level].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});
        body.append(tr);
        const [w,s,e,n]=zone.bounds;
        const rect=make('rect',{x:x(w),y:y(n),width:x(e)-x(w),height:y(s)-y(n),fill:habColors[prediction.risk_level],stroke:'#335c68',tabindex:0,role:'button','aria-label':`${zone.name}: ${prediction.risk_level}. Show input coverage.`});
        rect.addEventListener('click',()=>habDetail(zone));
        rect.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();habDetail(zone);}});
        svg.append(rect,make('text',{x:x(w)+8,y:y(n)+20},zone.name));
    });
    svg.append(make('text',{x:45,y:402},`${west.toFixed(2)}–${east.toFixed(2)}°E · ${south.toFixed(2)}–${north.toFixed(2)}°N`));
    habElement('hab-thresholds').textContent = `Moderate threshold: ${habForecast.thresholds.low_threshold.toFixed(3)}; High threshold: ${habForecast.thresholds.high_threshold.toFixed(3)}. Fitted on calibration observations; these categories are not harm severity levels.`;
}

function habCSV(data) {
    const escape = value => {
        let text = String(value ?? '');
        if (/^[=+@\-\t\r]/.test(text)) text="'"+text;
        return '"'+text.replaceAll('"','""')+'"';
    };
    const rows=[['forecast_date','day','zone_id','hab_occurrence_estimate','risk_category','input_end_date','chlorophyll_input_coverage_percent','freshness','historical_demo','operational_warning_available','lead_time_verified']];
    data.daily_predictions.forEach(day=>day.zones.forEach(zone=>rows.push([day.date,day.day,zone.zone_id,zone.probability,zone.risk_level,data.input_dates.at(-1),data.zones.find(z=>z.id===zone.zone_id).data_quality.chlorophyll_mean_input_coverage_percent,data.freshness.state,data.historical_demo,false,false])));
    return rows.map(row=>row.map(escape).join(',')).join('\r\n');
}

document.addEventListener('DOMContentLoaded',async()=>{
    habElement('hab-day').addEventListener('change',event=>habRender(Number(event.target.value)));
    habElement('hab-export').addEventListener('click',()=>{
        if(!habForecast)return;
        const url=URL.createObjectURL(new Blob([habCSV(habForecast)],{type:'text/csv;charset=utf-8'}));
        const link=document.createElement('a');link.href=url;link.download='hab-zone-forecast.csv';link.click();URL.revokeObjectURL(url);
    });
    habElement('hab-form').addEventListener('submit',async event=>{
        event.preventDefault();const request=++habRequest;habClear();habElement('hab-submit').disabled=true;
        habElement('hab-message').textContent='Checking input coverage and generating the HAB forecast…';
        try{
            const response=await fetch('/api/hab/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input_end_date:habElement('hab-date').value,historical_demo:habElement('hab-demo').checked}),signal:AbortSignal.timeout(120000)});
            const data=await response.json();if(request!==habRequest)return;
            if(!response.ok)throw new Error(data.detail||'Forecast unavailable.');
            if(!['success','insufficient_data'].includes(data.status))throw new Error(data.message||'HAB model is unavailable.');
            habForecast=data;
            habElement('hab-day').replaceChildren(...data.daily_predictions.map(row=>{const option=document.createElement('option');option.value=row.day;option.textContent=`Day +${row.day} · ${row.date}`;return option;}));
            habElement('hab-results').hidden=false;habRender(1);
            habElement('hab-message').textContent=data.status==='insufficient_data'?'Insufficient inputs: no model prediction was executed.':'HAB research forecast generated. External warnings remain disabled.';
        }catch(error){if(request===habRequest){habClear();habElement('hab-message').textContent=error.message;}}
        finally{if(request===habRequest)habElement('hab-submit').disabled=false;}
    });
    try{
        const response=await fetch('/api/hab/status',{signal:AbortSignal.timeout(15000)});
        if(!response.ok)throw new Error('Cannot read HAB model status.');
        const state=await response.json();habElement('hab-status').textContent=`${state.status}: ${state.message}`;
        habElement('hab-target').textContent=state.target;habElement('hab-submit').disabled=state.status!=='configured';
    }catch(error){habElement('hab-status').textContent=error.message;}
});
