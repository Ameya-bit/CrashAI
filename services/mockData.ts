
import { PredictionResult } from '../types.ts';

const FIXABLE_FEATURES: Record<string, string> = {
  ILLUMINATION: "Lighting Infrastructure",
  TCD_TYPE: "Traffic Control Devices",
  ROAD_CONDITION: "Road Surface Integrity",
  LANE_CLOSED: "Work Zone Protection",
  INTERSECT_TYPE: "Intersection Geometry",
  RDWY_SURF_TYPE_CD: "Pavement Friction",
  TCD_FUNC_CD: "Signal Hardware",
  WEATHER1: "Visibility Systems"
};

const RECOMMENDATIONS: Record<string, string> = {
  ILLUMINATION: "Upgrade to high-output LED fixtures and increase pole density for uniform coverage.",
  TCD_TYPE: "Implement high-visibility retroreflective signage and overhead electronic messaging.",
  ROAD_CONDITION: "Schedule immediate milling and resurfacing to correct surface deficiencies.",
  LANE_CLOSED: "Enhance longitudinal buffers and optimize taper lengths for improved traffic transition.",
  INTERSECT_TYPE: "Reconfigure to a compact roundabout or introduce protected left-turn phases.",
  RDWY_SURF_TYPE_CD: "Apply High Friction Surface Treatment (HFST) to reduce wet-weather skidding.",
  TCD_FUNC_CD: "Modernize controllers and radar-based detection sensors for real-time optimization.",
  WEATHER1: "Deploy automated fog-warning systems and road weather information sensors (RWIS)."
};

// FULL 100-ROW DATASET FROM PENNDOT CSV
const DATA_SAMPLE = `CRN,DEC_LATITUDE,DEC_LONGITUDE,MAX_SEVERITY_LEVEL,ILLUMINATION,TCD_TYPE,ROAD_CONDITION,LANE_CLOSED,INTERSECT_TYPE,RDWY_SURF_TYPE_CD,TCD_FUNC_CD,WEATHER1
2023040552,40.307757,-75.357684,4,1,2,1,1,1,1,2,3
2023042970,39.92582,-75.290094,3,0,2,1,0,0,1,0,3
2024002586,41.0485,-76.923292,3,2,1,7,0,0,4,0,10
2024003164,41.034331,-76.132624,0,0,1,7,0,0,4,0,10
2024003574,40.141077,-75.569385,0,0,1,7,0,0,5,0,3
2024002119,39.969104,-75.297156,4,2,2,1,0,1,1,2,3
2024003942,40.172246,-80.206414,0,0,1,1,0,0,4,0,3
2024003000,40.308572,-76.807744,0,0,1,7,0,0,1,0,10
2024002369,40.123503,-74.868244,0,0,1,9,1,0,5,0,7
2024002582,40.941764,-79.962824,0,0,1,7,0,0,4,0,10
2024004936,40.29847,-75.921049,0,0,1,9,0,0,5,0,4
2024004888,40.49623,-79.939233,0,0,2,7,1,0,1,0,10
2024006025,40.687397,-75.21082,3,2,2,1,0,1,1,2,3
2024007111,40.364285,-77.344735,0,0,2,7,1,0,1,0,10
2024006238,41.421642,-75.607754,0,0,1,7,0,0,2,0,10
2024005651,40.298343,-76.015749,0,0,1,9,0,2,4,0,4
2024005412,39.999951,-77.028544,3,3,2,1,0,1,1,2,3
2024006046,40.312976,-78.890003,0,0,1,2,1,0,2,0,10
2024006151,40.18209,-75.917208,0,0,1,7,0,0,5,0,10
2024005438,40.777733,-76.228999,0,0,1,1,0,2,4,0,4
2024005105,40.868275,-75.299354,0,0,1,2,0,0,2,0,3
2024005220,40.894386,-78.712679,0,0,1,2,0,0,4,0,10
2024006398,40.401208,-76.491985,0,0,1,5,0,0,4,0,10
2024006464,39.904514,-75.224183,2,1,2,2,0,0,1,2,10
2024002880,41.120253,-76.795534,0,0,1,0,0,0,1,0,3
2024006565,40.354845,-76.399774,2,2,1,1,0,0,1,2,3
2024000310,39.944746,-75.327807,8,1,0,0,0,0,1,0,3
2024005726,40.631334,-75.441383,4,1,1,4,0,0,1,2,4
2024006193,40.605308,-75.302041,0,0,1,0,0,0,2,0,3
2024003476,41.29723,-75.536073,8,1,1,2,0,0,4,1,3
2024002238,40.610438,-77.559505,0,0,2,2,0,0,1,0,10
2024006442,39.926624,-75.194663,8,2,2,4,0,0,1,2,10
2024004557,40.088438,-76.380604,3,3,2,3,0,0,1,2,3
2024006207,40.133989,-79.922989,0,0,1,2,0,0,4,0,3
2024003208,40.033539,-75.141322,9,0,2,0,0,0,1,0,10
2024006788,40.890446,-77.73349,0,0,2,0,0,0,1,0,3
2024004257,40.337064,-75.953565,3,1,1,1,0,0,1,1,3
2024000999,41.145412,-75.916676,2,1,1,6,0,0,1,0,3
2024004890,40.578011,-75.748824,0,0,1,0,0,0,4,0,10
2024004439,39.968409,-77.579199,0,0,1,1,0,0,4,0,10
2024002424,40.655479,-77.472878,0,0,1,0,0,0,4,0,3
2024003629,40.469432,-79.66803,0,0,1,6,0,0,5,0,9
2024000320,40.051022,-76.297159,0,0,5,3,0,0,1,3,4
2024004600,40.147089,-75.110385,3,1,2,5,0,0,1,3,7
2024003761,40.303098,-75.147853,4,1,3,0,2,0,1,4,7
2024001914,41.159319,-77.346677,0,0,1,0,0,0,1,0,3
2024004515,40.671068,-80.309634,0,0,1,0,0,1,1,3,4
2024003497,40.090945,-77.862336,0,0,1,0,0,0,4,0,10
2024004443,41.179819,-79.343091,4,1,1,4,0,0,4,0,2
2024004436,40.002415,-78.108943,3,1,1,5,7,0,1,0,3
2024003673,40.403961,-76.57746,0,0,1,0,0,0,1,1,3
2024004339,39.792706,-77.56179,3,3,1,5,0,0,1,3,7
2024002569,40.009647,-77.541143,0,0,1,1,0,0,4,0,10
2024006794,40.244006,-75.613594,2,1,1,2,0,0,1,2,3
2024003688,40.0023,-76.737608,0,0,1,0,0,0,4,0,10
2024006336,40.230294,-78.933987,4,1,1,2,0,0,1,0,10
2024002551,39.99025,-77.991142,3,1,1,3,0,0,4,0,10
2024006320,40.128298,-76.248243,0,0,2,0,0,0,5,0,3
2024002798,40.881827,-75.309909,3,2,1,2,0,0,1,0,3
2024006729,40.997569,-76.616799,3,1,2,0,0,0,2,0,3
2024002672,40.092789,-74.942354,4,1,1,4,0,0,1,2,3
2024003318,40.040249,-76.329886,8,1,1,0,0,0,1,3,3
2024001333,40.033002,-75.326394,3,1,1,3,0,0,1,0,3
2024004194,39.957253,-75.157893,0,0,1,3,0,0,1,0,3
2024006787,40.221101,-74.88049,0,0,1,4,0,0,1,0,3
2024003948,40.063403,-75.276003,0,0,1,4,0,0,1,0,3
2024002851,39.796246,-75.985994,3,1,1,0,0,0,4,0,6
2024004202,40.938403,-76.038178,0,0,1,0,0,0,1,2,10
2024004065,40.323331,-75.92921,9,0,7,0,0,0,1,2,3
2024003246,40.055342,-75.159347,3,1,1,0,7,0,1,0,3
2024004912,40.353662,-79.562109,0,0,1,2,0,0,5,0,3
2024004527,40.664872,-75.295144,0,0,1,0,7,0,1,0,3
2024000571,40.791761,-78.232826,0,0,1,5,0,0,4,0,2
2024005752,40.639297,-75.33374,0,0,3,0,0,0,1,0,3
2024006499,40.362595,-80.110476,0,0,2,0,0,0,1,2,3
2024004343,41.015432,-76.168299,0,0,1,0,0,0,3,0,7
2024002861,40.232334,-76.928431,0,0,7,0,2,0,2,0,7
2024002562,40.854886,-80.106101,0,0,1,0,0,0,4,0,2
2024005321,39.759084,-77.093468,3,3,1,0,0,0,4,0,10
2024005819,40.674741,-76.184662,3,1,1,0,0,0,4,0,3
2024002986,40.311078,-76.844386,0,0,1,1,0,0,4,0,3
2024002376,41.417719,-75.665849,0,0,2,0,0,0,4,2,3
2024003033,40.005826,-76.368417,0,0,1,0,0,0,4,0,10
2024002877,40.330492,-76.752507,3,4,1,4,0,0,1,2,3
2024006972,40.041496,-76.43892,0,0,4,0,0,0,1,0,10
2024003142,40.235169,-76.297772,0,0,1,0,0,0,1,0,3
2024003689,41.252654,-76.864617,3,1,1,0,0,0,4,0,10
2024006851,42.010588,-80.225164,3,2,1,0,0,0,4,0,6
2024002826,40.006367,-79.058126,0,0,1,4,0,0,1,0,10
2024000071,40.965716,-75.974186,0,0,1,0,0,0,1,3,3
2024006740,40.287117,-79.583451,3,1,1,3,0,0,1,0,3
2024003945,40.272936,-79.522492,3,1,1,6,0,0,2,0,6
2024003778,40.052242,-76.324772,3,1,1,0,0,0,1,0,7
2024001788,40.625641,-75.190843,3,2,1,5,0,0,1,0,3
2024004902,39.992353,-79.862467,0,0,1,5,0,0,2,0,6
2024004346,40.640323,-75.433365,3,3,1,1,0,0,1,2,7
2024002886,40.051484,-74.989335,8,1,1,2,0,0,1,0,3
2024001695,40.029379,-76.515076,0,0,1,0,3,0,1,0,3
2024007366,40.584539,-75.626384,8,1,1,0,0,0,1,2,3
2024003941,40.286067,-79.436695,0,0,1,5,0,0,4,0,3`;

let cachedData: any[] = [];

const parseCSV = (text: string) => {
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const obj: any = {};
    headers.forEach((h, i) => { obj[h] = values[i]; });
    return obj;
  });
};

export const initEngine = async (): Promise<any[]> => {
  if (cachedData.length === 0) {
    cachedData = parseCSV(DATA_SAMPLE);
  }
  return cachedData;
};

const getDistance = (lat1: number, lon1: number, lat2: number, lon2: number) => {
  const R = 6371; // km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon/2) * Math.sin(dLon/2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
};

export const predictRisk = async (lat: number, lng: number): Promise<PredictionResult> => {
  const data = await initEngine();
  await new Promise(r => setTimeout(r, 450));

  let weightedSeverity = 0;
  let totalWeight = 0;
  const p = 3.8; 
  const smoothing = 0.8; 

  data.forEach(d => {
    const dist = getDistance(lat, lng, parseFloat(d.DEC_LATITUDE), parseFloat(d.DEC_LONGITUDE));
    if (dist < 120) {
      const weight = 1 / Math.pow(dist + smoothing, p);
      const severity = parseInt(d.MAX_SEVERITY_LEVEL) || 0;
      weightedSeverity += severity * weight;
      totalWeight += weight;
    }
  });

  const proximityThreshold = 0.08; 
  const proximityFactor = Math.min(1.0, totalWeight / proximityThreshold);
  const rawScore = totalWeight > 0 ? (weightedSeverity / totalWeight) : 0;
  
  const riskScore = 1.1 + ((rawScore * 0.9) * proximityFactor);

  const featureImpacts = Object.keys(FIXABLE_FEATURES).map(key => {
    let featureInfluence = 0;
    let featWeight = 0;
    
    data.forEach(d => {
      const dist = getDistance(lat, lng, parseFloat(d.DEC_LATITUDE), parseFloat(d.DEC_LONGITUDE));
      if (dist < 120) {
        const weight = 1 / Math.pow(dist + smoothing, p);
        const val = d[key];
        const isDeficit = val && val !== '0' && val !== 'N' && val !== '7' && val !== '10';
        if (isDeficit) {
          featureInfluence += weight;
        }
        featWeight += weight;
      }
    });

    const impact = featWeight > 0 ? (featureInfluence / featWeight) * 4.5 : 0;
    return {
      feature: FIXABLE_FEATURES[key],
      impact: parseFloat((impact * proximityFactor + (Math.random() * 0.08)).toFixed(2)),
      recommendation: RECOMMENDATIONS[key]
    };
  });

  return {
    riskScore: parseFloat(Math.min(5.0, riskScore).toFixed(1)),
    topFixes: featureImpacts.sort((a, b) => b.impact - a.impact).slice(0, 3)
  };
};
