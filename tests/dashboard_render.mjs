// Render actual API JSON through existing dashboard functions in a minimal DOM.
// This verifies data binding and five-day switching, not browser layout.
import vm from 'node:vm';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
assert.equal(data.status, 'success');
const elements = new Map();
function element() {
  return {textContent:'',hidden:false,classList:{add(){},remove(){},toggle(){}},
    removeAttribute(name){delete this[name];},replaceChildren(){},append(){},
    insertRow(){return {append(){},insertCell(){return {};}};}};
}
const context = vm.createContext({console, Date,
  document:{getElementById(id){if(!elements.has(id)) elements.set(id,element());return elements.get(id);},
    querySelector(){return element();},querySelectorAll(){return Array.from({length:5},element);},
    createElement(){return element();},addEventListener(){}}});
vm.runInContext(fs.readFileSync('frontend/script.js','utf8'), context);
context.actualData = data;
vm.runInContext('predictionData = actualData; updateSystemInfo(actualData); updateForecastPeriod(); updateLatestInputDate();',context);
for (let day=1;day<=5;day++) {
  context.showDay(day);
  assert.equal(elements.get('prediction-map').src,data.maps[day-1]);
  assert.equal(elements.get('warning-level').textContent,data.daily_predictions[day-1].warning_level);
  assert.equal(elements.get('max-probability').textContent,`${(data.daily_predictions[day-1].max_probability*100).toFixed(1)}%`);
  assert.equal(elements.get('forecast-day').textContent,`Day +${day}`);
}
context.showUnavailable({status:'insufficient_data',message:'Contract check'});
assert.equal(elements.get('prediction-map').hidden,true);
assert.equal(elements.get('warning-level').textContent,'INSUFFICIENT DATA');
console.log('Actual backend data rendered for all five days; unavailable state hides maps.');
