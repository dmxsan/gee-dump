// Create your own geometry by clicking the polygon tool in the map
// Change the name based on the var training1 below

//Building Landsat image collection
function fmask(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  var qa = image.select('QA_PIXEL');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}

// Applies scaling factors
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

var composite764 = {
  bands: ['SR_B7', 'SR_B6', 'SR_B4'],
  min: 0.04,
  max: 0.35,
  gamma: 0.8,
};

var composite654 = {
  bands: ['SR_B6', 'SR_B5', 'SR_B4'],
  min: 0.022,
  max: 0.35,
  gamma: 0.48,
};

var composite562 = {
  bands: ['SR_B5', 'SR_B6', 'SR_B2'],
  min: 0.022,
  max: 0.35,
  gamma: 0.48,
};

//Landsat 8 2016
var image2 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterDate('2016-01-01', '2016-12-31')
                  .filterBounds(roi)
                  .map(fmask)
                  .map(applyScaleFactors);


var L8_2016 = image2.median().clip(roi);
Map.centerObject(roi, 12);
Map.addLayer(L8_2016, composite764, 'Landsat 8 2016 - 764');
Map.addLayer(L8_2016, composite654, 'Landsat 8 2016 - 654');
Map.addLayer(L8_2016, composite562, 'Landsat 8 2016 - 562');

var NDVI = L8_2016.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
var NDVIvisparam = {min: -1, max: 1, palette:['red', 'yellow', 'green']};
//Map.addLayer(NDVI, NDVIvisparam, 'NDVI');


//------------- Run Random Classification ---------------\\
var training1 = bush.merge(shrub).merge(other).merge(water);
var stacked_img = ee.Image.cat(L8_2016, NDVI);

var bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDVI']
var training1 = stacked_img.select(bands).sampleRegions({
  collection: training1,
  properties: ['lc'],
  scale: 30
});

var withRandom = training1.randomColumn('random');
var split = 0.8;
var trainingSample = withRandom.filter(ee.Filter.lt('random', split));
var testingSample = withRandom.filter(ee.Filter.gte('random', split));

var rf = ee.Classifier.smileRandomForest(500).train({
 features: trainingSample,
 classProperty: 'lc',
 inputProperties: bands
});
print(rf, 'RF model');

var test = testingSample.classify(rf);
print(test, 'Test');

//Variable Importance
var dict = rf.explain();
var explainTitle = 'Explain Random Forest :';
print(explainTitle, dict);

var varImp = ee.Feature(null, ee.Dictionary(dict).get('importance'));
var chart =
ui.Chart.feature.byProperty(varImp)
.setChartType('ColumnChart')
.setOptions({
title: 'Variable Importance',
legend: {position: 'none'},
hAxis: {title: 'Bands'},
vAxis: {title: 'Importance'}
});
print(chart);

var classified = stacked_img.select(bands).classify(rf);

//Visualize Random Classification Result
Map.addLayer(classified, 
{min: 1, max:4, palette: ['b6ff85', '248710', 'f5ad42', '42eff5']},
'LC Classification');

//Model accuracy
print('RF error matrix: ', rf.confusionMatrix());
print('RF accuracy: ', rf.confusionMatrix().accuracy());

//Validation accuracy
var validationAccuracy = test.errorMatrix('lc', 'classification');
print('Validation error matrix', validationAccuracy);
print('Validation accuracy: ', validationAccuracy.accuracy());
