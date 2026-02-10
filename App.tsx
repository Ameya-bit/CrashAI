
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, Marker, useMapEvents, useMap, Circle, ZoomControl } from 'react-leaflet';
import L from 'leaflet';
import { predictRisk, initEngine } from './services/mockData.ts';
import { MapPosition, PredictionResult } from './types.ts';

const TacticalPin = L.divIcon({
  className: 'custom-pin',
  html: `
    <div style="position: relative; width: 40px; height: 40px;">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2C8.13 2 5 5.13 5 9C5 14.25 12 22 12 22C12 22 19 14.25 19 9C19 5.13 15.87 2 12 2Z" fill="#1a73e8" stroke="white" stroke-width="2"/>
        <circle cx="12" cy="9" r="3" fill="white"/>
      </svg>
    </div>
  `,
  iconSize: [40, 40],
  iconAnchor: [20, 40],
});

const App: React.FC = () => {
  const [position, setPosition] = useState<MapPosition>({ lat: 40.3077, lng: -75.3576 });
  const [hoverPos, setHoverPos] = useState<MapPosition>({ lat: 40.3077, lng: -75.3576 });
  const [loading, setLoading] = useState(false);
  const [engineReady, setEngineReady] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [historicalPoints, setHistoricalPoints] = useState<any[]>([]);
  const [showInspector, setShowInspector] = useState(false);
  const [isSatellite, setIsSatellite] = useState(false);
  const [targetingActive, setTargetingActive] = useState(true);
  const [toast, setToast] = useState<string | null>(null);
  const [searchVal, setSearchVal] = useState("");

  const SIDEBAR_WIDTH = 440;

  useEffect(() => {
    const load = async () => {
      const data = await initEngine();
      setHistoricalPoints(data.map((d: any) => ({
        lat: parseFloat(d.DEC_LATITUDE),
        lng: parseFloat(d.DEC_LONGITUDE),
        severity: parseInt(d.MAX_SEVERITY_LEVEL) || 0,
      })).filter(d => !isNaN(d.lat) && !isNaN(d.lng)));
      setEngineReady(true);
    };
    load();
  }, []);

  const handleAnalysis = useCallback(async (lat: number, lng: number) => {
    if (!engineReady || !targetingActive) return;
    setLoading(true);
    setShowInspector(true);
    setResult(null);
    try {
      const prediction = await predictRisk(lat, lng);
      setResult(prediction);
    } catch (err) {
      console.error("Analysis failed", err);
    } finally {
      setLoading(false);
    }
  }, [engineReady, targetingActive]);

  const MapEvents = () => {
    useMapEvents({
      click(e) {
        if (!targetingActive) return;
        const newPos = { lat: e.latlng.lat, lng: e.latlng.lng };
        setPosition(newPos);
        handleAnalysis(newPos.lat, newPos.lng);
      },
      mousemove(e) {
        setHoverPos({ lat: e.latlng.lat, lng: e.latlng.lng });
      }
    });
    return null;
  };

  const ChangeView = ({ center }: { center: MapPosition }) => {
    const map = useMap();
    useEffect(() => {
      // Force Leaflet to recalculate container size
      setTimeout(() => map.invalidateSize(), 100);
    }, [center, map]);
    return null;
  };

  const jumpTo = (lat: number, lng: number) => {
    setPosition({ lat, lng });
    handleAnalysis(lat, lng);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const parts = searchVal.split(',').map(p => parseFloat(p.trim()));
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      jumpTo(parts[0], parts[1]);
      setToast("Targeting coordinates...");
    } else {
      setToast("Use Lat, Lng (e.g. 40.1, -75.1)");
    }
  };

  const getRiskColor = (score: number) => {
    if (score > 3.5) return '#d93025';
    if (score > 2) return '#f9ab00';
    return '#1a73e8';
  };

  return (
    <div className={`h-screen w-screen relative overflow-hidden bg-[#f1f3f4] text-[#202124] ${targetingActive ? 'targeting-mode' : ''}`}>

      {/* 1. Fluid Map Canvas */}
      <div className="absolute inset-0 z-0">
        <MapContainer
          center={[position.lat, position.lng]}
          zoom={14}
          className="h-full w-full"
          zoomControl={false}
          scrollWheelZoom={true}
          dragging={true}
        >
          <TileLayer
            url={isSatellite
              ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              : "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"}
          />

          {historicalPoints.map((point, i) => (
            <Circle
              key={i}
              center={[point.lat, point.lng]}
              radius={70}
              pathOptions={{
                fillColor: point.severity >= 3 ? '#d93025' : '#1a73e8',
                fillOpacity: 0.12,
                color: 'transparent',
                weight: 0
              }}
            />
          ))}

          <Marker position={[position.lat, position.lng]} icon={TacticalPin} />
          <MapEvents />
          <ChangeView center={position} />
          <ZoomControl position="bottomright" />
        </MapContainer>
      </div>

      {/* 2. Tactical Command Pill (Header) */}
      <div
        className={`absolute top-8 z-[1001] ui-transition pointer-events-none`}
        style={{ left: showInspector ? `${SIDEBAR_WIDTH + 32}px` : '32px' }}
      >
        <div className="bg-white/95 backdrop-blur-xl rounded-full shadow-2xl pill-shadow flex items-center p-1.5 pointer-events-auto h-16 border border-white/50">

          {/* Status & Toggle Segment */}
          <div className="flex items-center gap-4 pl-5 pr-6 border-r border-gray-100">
            <div className="w-10 h-10 bg-[#1a73e8] rounded-full flex items-center justify-center text-white font-bold">C</div>
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-tight text-[#1a73e8]">CRASHAI</span>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${targetingActive ? 'bg-emerald-500 animate-pulse' : 'bg-gray-300'}`}></div>
                <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">{targetingActive ? 'Active Scan' : 'Ready'}</span>
              </div>
            </div>
          </div>

          {/* Targeting Toggle Segment */}
          <div className="px-6 flex items-center gap-3 border-r border-gray-100 h-full">
            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Targeting</span>
            <button
              onClick={() => setTargetingActive(!targetingActive)}
              className={`w-12 h-6 rounded-full relative transition-colors duration-300 ${targetingActive ? 'bg-emerald-500' : 'bg-gray-200'}`}
            >
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all duration-300 ${targetingActive ? 'left-7' : 'left-1'}`}></div>
            </button>
          </div>

          {/* Search/Jump Input */}
          <form onSubmit={handleSearch} className="flex items-center pl-3 pr-2">
            <input
              type="text"
              placeholder="Lat, Lng jump..."
              value={searchVal}
              onChange={(e) => setSearchVal(e.target.value)}
              className="bg-transparent text-sm font-medium px-4 py-2 w-44 focus:outline-none placeholder:text-gray-300"
            />
            <button type="submit" className="w-11 h-11 rounded-full bg-blue-50 hover:bg-blue-100 flex items-center justify-center text-[#1a73e8] transition-all">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" /></svg>
            </button>
          </form>
        </div>
      </div>

      {/* 3. Global Control Pills (Bottom Left) */}
      <div
        className={`absolute bottom-10 z-[1001] ui-transition flex items-center gap-4`}
        style={{ left: showInspector ? `${SIDEBAR_WIDTH + 32}px` : '32px' }}
      >
        <button
          onClick={() => setIsSatellite(!isSatellite)}
          className="bg-white rounded-full px-6 py-4 shadow-2xl pill-shadow flex items-center gap-3 hover:bg-[#f8f9fa] transition-all group border border-white/80"
        >
          <div className="w-9 h-9 rounded-full overflow-hidden border-2 border-white shadow-inner">
            <img
              src={isSatellite
                ? "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/14/4678/6055.png"
                : "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/14/6055/4678"}
              className="w-full h-full object-cover"
            />
          </div>
          <span className="text-xs font-bold text-[#1a73e8] uppercase tracking-widest">
            {isSatellite ? 'Street' : 'Satellite'}
          </span>
        </button>

        <div className="bg-white/80 backdrop-blur-md rounded-full px-6 py-4 shadow-xl border border-white/50 text-[10px] font-bold text-gray-500 flex items-center gap-4">
          <span className="font-mono text-[#1a73e8]">{hoverPos.lat.toFixed(5)}</span>
          <span className="font-mono text-[#1a73e8]">{hoverPos.lng.toFixed(5)}</span>
        </div>
      </div>

      {/* 4. Tactical Sidebar (Ultra Rounded) */}
      <aside
        className={`absolute top-0 left-0 z-[1002] transition-transform duration-[600ms] ease-[cubic-bezier(0.34, 1.56, 0.64, 1)] h-full bg-white shadow-[0_30px_60px_rgba(0,0,0,0.18)] flex flex-col overflow-hidden rounded-r-[4rem]`}
        style={{ width: `${SIDEBAR_WIDTH}px`, transform: showInspector ? 'translateX(0)' : 'translateX(-105%)' }}
      >
        {/* Cover Section */}
        <div className="relative h-72 flex-shrink-0">
          <img
            src={`https://images.unsplash.com/photo-1545143333-11be56138b9d?auto=format&fit=crop&q=80&w=800`}
            className="w-full h-full object-cover"
            alt="Infrastructure"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-white via-transparent to-transparent"></div>
          <button
            onClick={() => setShowInspector(false)}
            className="absolute top-8 left-8 bg-white shadow-xl rounded-full p-4 text-[#1a73e8] hover:scale-110 active:scale-95 transition-all"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" /></svg>
          </button>
        </div>

        {/* Dynamic Content */}
        <div className="flex-1 overflow-y-auto custom-scroll px-10 pb-10">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full gap-6">
              <div className="w-16 h-16 border-[5px] border-blue-50 border-t-[#1a73e8] rounded-full animate-spin"></div>
              <span className="text-xs font-bold text-[#1a73e8] uppercase tracking-[0.2em] animate-pulse">Running Neural Capture</span>
            </div>
          ) : result ? (
            <div className="space-y-10">
              {/* Identity Header */}
              <div>
                <h2 className="text-3xl font-bold tracking-tight mb-1">Site Analysis</h2>
                <p className="text-xs font-bold text-gray-400 uppercase tracking-widest">ID {Math.floor(position.lat * 100000)} · Pennsylvania</p>
              </div>

              {/* Safety Rating Pill */}
              <div
                className="rounded-[3rem] p-8 text-white pill-shadow flex items-center justify-between"
                style={{ backgroundColor: getRiskColor(result.riskScore) }}
              >
                <div className="flex flex-col">
                  <span className="text-xs font-bold uppercase tracking-widest opacity-80 mb-1">Crash Propensity</span>
                  <span className="text-5xl font-bold">{result.riskScore}</span>
                </div>
                <div className="bg-white/20 backdrop-blur-md px-6 py-2 rounded-full">
                  <span className="text-xs font-bold uppercase tracking-widest">{result.riskScore > 3.5 ? 'Critical' : 'Nominal'}</span>
                </div>
              </div>

              {/* Severity Probabilities */}
              {result.severityProbs && (
                <div className="space-y-4">
                  <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 pl-4">Severity Probabilities</h3>
                  <div className="bg-gray-50 rounded-[2.5rem] p-6 border border-gray-100/50 space-y-3">
                    {result.severityProbs.map((sp, idx) => (
                      <div key={idx} className="flex items-center gap-3">
                        <span className="text-[10px] font-bold text-gray-500 w-28 text-right truncate">{sp.label}</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.max(2, sp.probability * 100)}%`,
                              backgroundColor: sp.level >= 4 ? '#d93025' : sp.level >= 3 ? '#f9ab00' : '#1a73e8'
                            }}
                          ></div>
                        </div>
                        <span className="text-[10px] font-bold text-gray-600 w-12">{(sp.probability * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tactical Actions */}
              <div className="flex gap-4">
                <button onClick={() => setToast("Safety protocols deployed.")} className="flex-1 h-14 rounded-full bg-[#1a73e8] text-white text-xs font-bold shadow-xl hover:shadow-blue-500/40 transition-all uppercase tracking-widest">Deploy Fixes</button>
                <button className="w-14 h-14 rounded-full bg-gray-50 flex items-center justify-center text-gray-400 hover:text-[#1a73e8] transition-colors"><svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M17 3H7c-1.1 0-1.99.9-1.99 2L5 21l7-3 7 3V5c0-1.1-.9-2-2-2z" /></svg></button>
              </div>

              {/* Fix List */}
              <div className="space-y-5">
                <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 pl-4">Priority Interventions</h3>
                <div className="space-y-4">
                  {result.topFixes.map((fix, idx) => (
                    <div key={idx} className="bg-gray-50 rounded-[2.5rem] p-6 border border-gray-100/50 hover:bg-white hover:border-[#1a73e8]/20 transition-all cursor-default group">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-bold text-gray-800">{fix.feature}</span>
                        <span className="text-[10px] font-bold text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full">-{fix.impact} pts</span>
                      </div>
                      <div className="mb-3">
                        <span className="text-[10px] font-bold text-orange-600 bg-orange-50 px-3 py-1 rounded-full">Current: {fix.currentValue}</span>
                      </div>
                      <p className="text-xs text-gray-500 leading-relaxed font-medium">{fix.recommendation}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center p-8">
              <div className="w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mb-6">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="#1a73e8"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z" /></svg>
              </div>
              <h4 className="text-lg font-bold mb-2">Initialize Scanner</h4>
              <p className="text-sm text-gray-400 font-medium">Toggle "Targeting" on the top pill and click any intersection to capture safety data.</p>
            </div>
          )}
        </div>

        {/* Footer Pill */}
        <div className="px-10 py-8 bg-gray-50 flex justify-between items-center border-t border-gray-100">
          <div className="flex flex-col">
            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Protocol</span>
            <span className="text-[11px] font-bold text-[#1a73e8]">PennDOT Secure-v4</span>
          </div>
          <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
        </div>
      </aside>

      {/* 5. Tactical Toasts */}
      {toast && (
        <div className="fixed bottom-12 left-1/2 -translate-x-1/2 z-[2000] animate-in slide-in-from-bottom-8 fade-in duration-500">
          <div className="bg-[#1a1a1a] text-white px-10 py-5 rounded-full shadow-2xl pill-shadow text-xs font-bold tracking-[0.15em] flex items-center gap-8 border border-white/10 uppercase">
            <span className="flex-1">{toast}</span>
            <button onClick={() => setToast(null)} className="text-[#1a73e8] hover:text-white transition-colors">Dismiss</button>
          </div>
        </div>
      )}

      <style>{`
        .leaflet-marker-icon { transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1); }
        .custom-pin svg { filter: drop-shadow(0 10px 15px rgba(26,115,232,0.3)); transition: all 0.3s ease; }
        .custom-pin:hover svg { transform: scale(1.1) translateY(-5px); }
      `}</style>
    </div>
  );
};

export default App;
