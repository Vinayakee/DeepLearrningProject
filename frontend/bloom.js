/* Satellite bloom-event forecast. Actual inference only; no proxy-to-HAB fallback. */
let bloomData=null;
let bloomSerial=0;
const bloomElement=id=>document.getElementById(id);
function bloomClear(){bloomData=null;bloomElement('bloom-results').hidden=true;bloomElement('bloom-map').removeAttribute('src');}
function bloomDay(day){
    if(!bloomData||day<1||day>5)return;
    const row=bloomData.daily_predictions[day-1];
    bloomElement('bloom-map').src=bloomData.maps[day-1];
    bloomElement('bloom-context').textContent=`Day +${day}: ${row.date}. Inputs ${bloomData.input_dates[0]} to ${bloomData.input_dates.at(-1)}. ${bloomData.historical_demo?'Historical demonstration.':''}`;
    const kind=bloomData.probability_calibrated?'2024 validation-calibrated score':'raw model score';
    bloomElement('bloom-score').textContent=`${row.event_pixels} event pixels / ${row.supported_pixels} supported pixels. Maximum ${kind} ${(row.max_score*100).toFixed(2)}%. Threshold ${bloomData.threshold}. Research development fit; no operational confidence guarantee.`;
}
function bloomRender(data){
    bloomData=data;
    bloomElement('bloom-day').replaceChildren(...data.daily_predictions.map(row=>{const node=document.createElement('option');node.value=row.day;node.textContent=`Day +${row.day} · ${row.date}`;return node;}));
    bloomElement('bloom-table').replaceChildren(...data.daily_predictions.map(row=>{const tr=document.createElement('tr');[row.date,row.event_pixels,row.supported_pixels,`${(row.max_score*100).toFixed(2)}%`].forEach(value=>{const td=document.createElement('td');td.textContent=String(value);tr.append(td);});return tr;}));
    bloomElement('bloom-export').href=`/api/bloom/forecasts/${encodeURIComponent(data.forecast_id)}/summary.csv?demo=${data.historical_demo}`;
    bloomElement('bloom-score-kind').textContent=data.probability_calibrated?'Maximum validation-calibrated score':'Maximum raw score';
    bloomElement('bloom-coverage').textContent=`Mean chlorophyll input coverage: ${data.data_quality.chlorophyll_mean_input_coverage_percent.toFixed(2)}%; freshness: ${data.freshness.state}. ${data.validation_limit}`;
    bloomElement('bloom-results').hidden=false;bloomDay(1);
}
async function bloomStatus(){
    const serial=++bloomSerial;
    const run=bloomElement('bloom-run').value;
    bloomElement('bloom-submit').disabled=true;
    bloomElement('bloom-status').textContent='Checking selected experiment…';
    bloomElement('bloom-evaluation').href=`/api/bloom/evaluation?run_id=${encodeURIComponent(run)}`;
    try{
        const response=await fetch(`/api/bloom/status?run_id=${encodeURIComponent(run)}`,{signal:AbortSignal.timeout(15000)});
        if(!response.ok){
            if(response.status===404||response.status===422)throw new Error('This server lacks the selected experiment API. Restart from the updated project using the README launch command (port 8020).');
            throw new Error(`Model status unavailable (HTTP ${response.status}). Check the server log.`);
        }
        const data=await response.json();if(serial!==bloomSerial)return;
        if(data.run_id!==run)throw new Error('This server does not support the selected run. Launch the updated app on a free port.');
        bloomElement('bloom-status').textContent=`${data.run_id} · ${data.status}: ${data.message}`;
        bloomElement('bloom-definition').textContent=data.label_definition||'';
        bloomElement('bloom-submit').disabled=data.status!=='ready_research_only';
    }catch(error){if(serial===bloomSerial)bloomElement('bloom-status').textContent=error.message;}
}
document.addEventListener('DOMContentLoaded',async()=>{
    bloomElement('bloom-run').addEventListener('change',()=>{bloomClear();bloomElement('bloom-message').textContent='Experiment changed; generate a new forecast.';bloomStatus();});
    bloomElement('bloom-day').addEventListener('change',event=>bloomDay(Number(event.target.value)));
    bloomElement('bloom-map').addEventListener('error',()=>{bloomClear();bloomElement('bloom-message').textContent='Forecast map unavailable. Regenerate the forecast.';});
    bloomElement('bloom-form').addEventListener('submit',async event=>{
        event.preventDefault();const serial=++bloomSerial;const run=bloomElement('bloom-run').value;bloomClear();bloomElement('bloom-submit').disabled=true;
        bloomElement('bloom-message').textContent='Checking real input observations and generating five maps…';
        try{
            const response=await fetch('/api/bloom/predict',{method:'POST',headers:{'Content-Type':'application/json'},signal:AbortSignal.timeout(120000),body:JSON.stringify({input_end_date:bloomElement('bloom-date').value,historical_demo:bloomElement('bloom-demo').checked,run_id:run})});
            const data=await response.json();
            if(serial!==bloomSerial)return;
            if(!response.ok)throw new Error(data.detail||'Forecast request failed.');
            if(data.status!=='success'){bloomElement('bloom-message').textContent=data.status==='insufficient_data'?'Insufficient observed inputs; no model inference executed.':data.message;return;}
            bloomRender(data);bloomElement('bloom-message').textContent='Actual satellite bloom-event forecast generated. Research only; no confirmed HAB claim.';
        }catch(error){if(serial===bloomSerial){bloomClear();bloomElement('bloom-message').textContent=error.message;}}
        finally{if(serial===bloomSerial)bloomElement('bloom-submit').disabled=false;}
    });
    await bloomStatus();
});
