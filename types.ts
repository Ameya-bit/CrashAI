
export interface CrashData {
  CRN: string;
  DEC_LATITUDE: number;
  DEC_LONGITUDE: number;
  MAX_SEVERITY_LEVEL: number;
  ILLUMINATION: string;
  TCD_TYPE: string;
  ROAD_CONDITION: string;
  LANE_CLOSED: string;
  INTERSECT_TYPE: string;
  URBAN_RURAL: string;
  RDWY_SURF_TYPE_CD: string;
  WORK_ZONE_IND: string;
  TCD_FUNC_CD: string;
}

export interface PredictionResult {
  riskScore: number;
  topFixes: {
    feature: string;
    impact: number;
    recommendation: string;
  }[];
}

export interface MapPosition {
  lat: number;
  lng: number;
}
