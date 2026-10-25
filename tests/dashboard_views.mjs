// Exercise observation view, export routing and zoom with actual backend JSON.
import vm from 'node:vm';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const elements = new Map();
function element() {
  return {textContent:'',hidden:false,style:{},classList:{add(){},remove(){},toggle(){}},
    removeAttribute(name){delete this[name];},replaceChildren(){},append(){},
    insertRow(){return {append(){},insertCell(){return {};}};}};
}
const context = vm.createContext({console, Date, encodeURIComponent,
  document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},
    querySelector(){return element();},querySelectorAll(){return Array.from({length:5},element);},
    createElement(){return element();},addEventListener(){}}});
vm.runInContext(fs.readFileSync('frontend/script.js','utf8'), context);
context.actualData = data;
vm.runInContext('predictionData = actualData; demoMode = true; updateSystemInfo(actualData); mapView = "observations"; mapZoom = 2; showDay(3);',context);
assert.equal(elements.get('prediction-map').src,`/api/forecasts/${data.forecast_id}/observations.png?demo=true`);
assert.equal(elements.get('prediction-map').style.width,'200%');
assert.equal(elements.get('forecast-export').href,`/api/forecasts/${data.forecast_id}/summary.csv?demo=true`);
assert.match(elements.get('map-context-note').textContent,/stays fixed/);
vm.runInContext('mapView = "forecast"; showDay(5);',context);
assert.equal(elements.get('prediction-map').src,data.maps[4]);
context.clearForecast('Unavailable');
assert.equal(elements.get('forecast-export').hidden,true);
assert.equal(elements.get('map-view').disabled,true);
assert.equal(elements.get('prediction-map').hidden,true);
console.log('Observation/forecast switching, zoom, export routing and unavailable state passed.');
