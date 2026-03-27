const data = window.CHENNAI_DEMO_DATA;

const vehicleProfiles = {
  compact: {
    id: "compact",
    label: "Compact EV",
    batteryKwh: 30,
    baseKwhPerKm: 0.138,
    regenFactor: 0.64
  },
  standard: {
    id: "standard",
    label: "Standard EV",
    batteryKwh: 40,
    baseKwhPerKm: 0.152,
    regenFactor: 0.60
  },
  suv: {
    id: "suv",
    label: "SUV EV",
    batteryKwh: 60,
    baseKwhPerKm: 0.186,
    regenFactor: 0.56
  }
};

const cabinProfiles = {
  eco: { id: "eco", label: "Eco cabin", loadKw: 0.25 },
  normal: { id: "normal", label: "Normal cabin", loadKw: 0.70 },
  hot: { id: "hot", label: "Hot Chennai AC", loadKw: 1.30 }
};

const driverProfiles = {
  calm: {
    id: "calm",
    label: "Calm driver",
    efficiencyMul: 0.95,
    durationFactor: 1.03,
    stopGoFactor: 0.93,
    speedDelta: -3,
    riskBias: -0.08,
    note: "Smoother throttle and braking. Better efficiency, slightly slower ETA."
  },
  normal: {
    id: "normal",
    label: "Normal driver",
    efficiencyMul: 1,
    durationFactor: 1,
    stopGoFactor: 1,
    speedDelta: 0,
    riskBias: 0,
    note: "Balanced real-world driving. Neutral reference profile."
  },
  aggressive: {
    id: "aggressive",
    label: "Aggressive driver",
    efficiencyMul: 1.08,
    durationFactor: 0.97,
    stopGoFactor: 1.14,
    speedDelta: 4,
    riskBias: 0.12,
    note: "Higher cruise speed and sharper stop-go behavior. Faster on paper, less battery-stable."
  }
};

const trafficProfiles = {
  normal: {
    id: "normal",
    label: "Normal flow",
    etaBoost: 0.06,
    slowAdd: 0.018,
    crawlAdd: 0.006,
    energyBias: 0,
    cabinBiasKw: 0,
    uncertaintyBias: 0,
    note: "Baseline conditions. Route shape and corridor smoothness dominate."
  },
  peak: {
    id: "peak",
    label: "Peak congestion",
    etaBoost: 0.24,
    slowAdd: 0.11,
    crawlAdd: 0.038,
    energyBias: 0.04,
    cabinBiasKw: 0.12,
    uncertaintyBias: 0.18,
    note: "Heavy stop-go traffic. Reliability becomes more important than nominal ETA."
  },
  hotday: {
    id: "hotday",
    label: "Hot afternoon",
    etaBoost: 0.1,
    slowAdd: 0.03,
    crawlAdd: 0.01,
    energyBias: 0.03,
    cabinBiasKw: 0.55,
    uncertaintyBias: 0.08,
    note: "Cabin load rises sharply. Range drops even if congestion is only moderate."
  },
  rain: {
    id: "rain",
    label: "Rain slowdown",
    etaBoost: 0.2,
    slowAdd: 0.09,
    crawlAdd: 0.032,
    energyBias: 0.05,
    cabinBiasKw: 0.15,
    uncertaintyBias: 0.2,
    note: "Wet roads and slower flow increase volatility. The safer corridor often wins."
  }
};

const colors = ["#d86d33", "#1a7b77", "#305181"];
const etaPenaltyPerMinute = 0.035;

const state = {
  presetId: "airport-iit",
  vehicleId: "standard",
  driverId: "normal",
  cabinId: "normal",
  trafficId: "normal",
  decisionValue: 50,
  soc: 35,
  selectedRouteId: null,
  lastRecommendedId: null,
  flipPulseTimer: null
};

const refs = {
  tripSelect: document.getElementById("trip-select"),
  vehicleSelect: document.getElementById("vehicle-select"),
  driverSelect: document.getElementById("driver-select"),
  trafficSelect: document.getElementById("traffic-select"),
  cabinSelect: document.getElementById("cabin-select"),
  decisionRange: document.getElementById("decision-range"),
  decisionValue: document.getElementById("decision-value"),
  decisionCopy: document.getElementById("decision-copy"),
  socRange: document.getElementById("soc-range"),
  socValue: document.getElementById("soc-value"),
  generatedPill: document.getElementById("generated-pill"),
  statusCopy: document.getElementById("status-copy"),
  reviewerCopy: document.getElementById("reviewer-copy"),
  formulaBreakdown: document.getElementById("formula-breakdown"),
  mapTitle: document.getElementById("map-title"),
  mapSubtitle: document.getElementById("map-subtitle"),
  mapStats: document.getElementById("map-stats"),
  routeMap: document.getElementById("route-map"),
  mapLegend: document.getElementById("map-legend"),
  comparisonGrid: document.getElementById("comparison-grid"),
  riskGrid: document.getElementById("risk-grid"),
  rationaleGrid: document.getElementById("rationale-grid"),
  energyGrid: document.getElementById("energy-grid"),
  summaryGrid: document.getElementById("summary-grid"),
  routesGrid: document.getElementById("routes-grid"),
  selectedTitle: document.getElementById("selected-title"),
  selectedCaption: document.getElementById("selected-caption"),
  reviewerNote: document.getElementById("reviewer-note"),
  creditLine: document.getElementById("credit-line"),
  flipCard: document.getElementById("flip-card"),
  flipHeadline: document.getElementById("flip-headline"),
  flipZone: document.getElementById("flip-zone"),
  flipThreshold: document.getElementById("flip-threshold"),
  flipCurrent: document.getElementById("flip-current"),
  flipCopy: document.getElementById("flip-copy")
};

function formatNumber(value, digits = 1) {
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits
  }).format(value);
}

function signedNumber(value, digits = 1) {
  const sign = value > 0 ? "+" : "";
  return sign + formatNumber(value, digits);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getPreset() {
  return data.presets.find((preset) => preset.id === state.presetId);
}

function getVehicle() {
  return vehicleProfiles[state.vehicleId];
}

function getDriver() {
  return driverProfiles[state.driverId];
}

function getCabin() {
  return cabinProfiles[state.cabinId];
}

function getTraffic() {
  return trafficProfiles[state.trafficId];
}

function getDecisionProfile(decisionValue = state.decisionValue) {
  const t = decisionValue / 100;

  if (t <= 0.2) {
    return {
      label: "Fastest first",
      key: "fast",
      etaWeight: 2.9,
      reserveScale: 0.06,
      energyWeight: 0.78,
      note: "ETA is heavily prioritized. Only strong battery gains can justify a detour."
    };
  }

  if (t >= 0.8) {
    return {
      label: "Safest arrival",
      key: "safe",
      etaWeight: 0.18,
      reserveScale: 2.25,
      energyWeight: 1.08,
      note: "Arrival reliability is prioritized. The optimizer will spend time to protect battery confidence."
    };
  }

  return {
    label: "Balanced tradeoff",
    key: "balanced",
    etaWeight: 1,
    reserveScale: 1,
    energyWeight: 1,
    note: "ETA, energy use, and reliability reserve all matter."
  };
}

function getReserveWeight(soc = state.soc, cabinId = state.cabinId) {
  const cabinBias = cabinId === "hot" ? 0.1 : 0;
  if (soc <= 25) {
    return 1.2 + cabinBias;
  }
  if (soc <= 45) {
    return 0.45 + cabinBias;
  }
  return 0.15 + cabinBias;
}

function getRouteExposure(route) {
  let corridorBias = 1;
  if (route.id.includes("short")) {
    corridorBias = 1.24;
  } else if (route.id.includes("smooth")) {
    corridorBias = 0.92;
  } else if (route.id.includes("safe")) {
    corridorBias = 0.82;
  }

  const pressure = 0.62 + (route.slowShare * 1.8) + (route.crawlShare * 2.7);
  return clamp(pressure * corridorBias, 0.65, 1.65);
}

function buildContext(overrides = {}) {
  const vehicle = overrides.vehicle || getVehicle();
  const driver = overrides.driver || getDriver();
  const cabin = overrides.cabin || getCabin();
  const traffic = overrides.traffic || getTraffic();
  const decision = overrides.decision || getDecisionProfile(overrides.decisionValue ?? state.decisionValue);
  const soc = overrides.soc ?? state.soc;
  const reserveWeight = overrides.reserveWeight ?? getReserveWeight(soc, cabin.id);
  const availableKwh = vehicle.batteryKwh * (soc / 100);

  return {
    vehicle,
    driver,
    cabin,
    traffic,
    decision,
    soc,
    reserveWeight,
    availableKwh
  };
}

function computeRouteMetrics(route, overrides = {}) {
  const context = buildContext(overrides);
  const exposure = getRouteExposure(route);
  const adjustedSlowShare = clamp(
    route.slowShare + (context.traffic.slowAdd * exposure),
    0.02,
    0.76
  );
  const adjustedCrawlShare = clamp(
    route.crawlShare + (context.traffic.crawlAdd * exposure),
    0.002,
    0.36
  );
  const scenarioDurationMin = route.durationMin * (1 + (context.traffic.etaBoost * exposure)) * context.driver.durationFactor;
  const effectiveAvgSpeed = Math.max(16, route.avgSpeedKmh + context.driver.speedDelta);
  const baseDriveKwh = route.distanceKm * context.vehicle.baseKwhPerKm * context.driver.efficiencyMul * (1 + context.traffic.energyBias);
  const congestionPenaltyKwh = baseDriveKwh * ((0.85 * adjustedSlowShare) + (0.55 * adjustedCrawlShare)) * context.driver.stopGoFactor;
  const speedPenaltyKwh = baseDriveKwh * (0.0016 * Math.pow(Math.max(effectiveAvgSpeed - 42, 0), 2));
  const climbKwh = route.ascentM * 0.004;
  const regenCreditKwh = route.descentM * context.vehicle.regenFactor * 0.002;
  const cabinKwh = (context.cabin.loadKw + context.traffic.cabinBiasKw) * (scenarioDurationMin / 60);

  let expectedKwh = baseDriveKwh + congestionPenaltyKwh + speedPenaltyKwh + climbKwh + cabinKwh - regenCreditKwh;
  const expectedFloor = baseDriveKwh * 0.88;
  let floorLiftKwh = 0;
  if (expectedKwh < expectedFloor) {
    floorLiftKwh = expectedFloor - expectedKwh;
    expectedKwh = expectedFloor;
  }

  const reserveMultiplier = 0.08 + (1.35 * adjustedSlowShare) + (0.95 * adjustedCrawlShare);
  const reserveKwh = expectedKwh * reserveMultiplier * (0.58 + (0.34 * exposure)) * (1 + context.traffic.uncertaintyBias + context.driver.riskBias);
  const robustKwh = expectedKwh + reserveKwh;
  const arrivalPct = ((context.availableKwh - expectedKwh) / context.vehicle.batteryKwh) * 100;
  const arrivalSafePct = ((context.availableKwh - robustKwh) / context.vehicle.batteryKwh) * 100;
  const uncertaintyGapPct = Math.max(0, arrivalPct - arrivalSafePct);
  const riskIndex = clamp(
    44 +
      ((exposure - 1) * 20) +
      (context.traffic.uncertaintyBias * 120) +
      (context.driver.riskBias * 55) +
      (Math.max(0, 8 - arrivalSafePct) * 3.2) +
      (uncertaintyGapPct * 1.5) -
      (Math.max(arrivalSafePct, 0) * 1.12),
    4,
    92
  );
  const successProbabilityPct = clamp(100 - riskIndex, 8, 98);
  const riskBandLowPct = Math.min(arrivalSafePct, arrivalPct);
  const riskBandHighPct = clamp(arrivalPct + (uncertaintyGapPct * 0.35), arrivalPct, 100);
  let riskLevel = "Low risk";
  if (successProbabilityPct < 70) {
    riskLevel = "High risk";
  } else if (successProbabilityPct < 85) {
    riskLevel = "Medium risk";
  }

  return {
    scenarioDurationMin,
    adjustedSlowShare,
    adjustedCrawlShare,
    exposure,
    expectedKwh,
    reserveKwh,
    robustKwh,
    arrivalPct,
    arrivalSafePct,
    riskIndex,
    successProbabilityPct,
    riskBandLowPct,
    riskBandHighPct,
    riskLevel,
    components: {
      baseDriveKwh: baseDriveKwh + floorLiftKwh,
      congestionPenaltyKwh,
      speedPenaltyKwh,
      climbKwh,
      cabinKwh,
      regenCreditKwh
    }
  };
}

function annotateRoutes(preset, overrides = {}) {
  const context = buildContext(overrides);
  const annotated = preset.routes.map((route, index) => ({
    ...route,
    color: colors[index % colors.length],
    metrics: computeRouteMetrics(route, overrides)
  }));

  const fastest = annotated.reduce((best, route) => {
    return route.metrics.scenarioDurationMin < best.metrics.scenarioDurationMin ? route : best;
  }, annotated[0]);
  const lowestEnergy = annotated.reduce((best, route) => {
    return route.metrics.expectedKwh < best.metrics.expectedKwh ? route : best;
  }, annotated[0]);
  const safest = annotated.reduce((best, route) => {
    return route.metrics.successProbabilityPct > best.metrics.successProbabilityPct ? route : best;
  }, annotated[0]);

  annotated.forEach((route) => {
    route.etaPenaltyKwh = Math.max(0, route.metrics.scenarioDurationMin - fastest.metrics.scenarioDurationMin) * etaPenaltyPerMinute;
    route.riskPenalty = (100 - route.metrics.successProbabilityPct) * 0.018 * context.decision.reserveScale;
    route.policyScore =
      (route.metrics.expectedKwh * context.decision.energyWeight) +
      (route.metrics.reserveKwh * context.reserveWeight * context.decision.reserveScale) +
      (route.etaPenaltyKwh * context.decision.etaWeight) +
      route.riskPenalty;
    route.badges = [];

    if (route.id === fastest.id) {
      route.badges.push({ label: "Fastest", type: "fastest" });
    }
    if (route.id === lowestEnergy.id) {
      route.badges.push({ label: "Lowest energy", type: "energy" });
    }
    if (route.id === safest.id) {
      route.badges.push({ label: "Highest confidence", type: "safe" });
    }
  });

  const recommended = annotated.reduce((best, route) => {
    return route.policyScore < best.policyScore ? route : best;
  }, annotated[0]);

  return { annotated, recommended, fastest };
}

function isConfidenceMode(annotated, overrides = {}) {
  const context = buildContext(overrides);
  const minimumRobust = Math.min(...annotated.map((route) => route.metrics.robustKwh));
  return (
    context.soc <= 30 ||
    context.cabin.id === "hot" ||
    context.traffic.id === "peak" ||
    context.traffic.id === "rain" ||
    context.decision.key === "safe" ||
    context.availableKwh < (minimumRobust + (context.vehicle.batteryKwh * 0.08))
  );
}

function deltaClass(value, positiveIsGood = true) {
  if (Math.abs(value) < 0.05) {
    return "neutral";
  }

  const isGood = positiveIsGood ? value > 0 : value < 0;
  return isGood ? "positive" : "warning";
}

function buildStatusCopy(preset, recommended, fastest, confidenceMode) {
  const traffic = getTraffic();
  const decision = getDecisionProfile();
  if (recommended.id === fastest.id) {
    return `${traffic.label}: under ${decision.label.toLowerCase()}, the optimizer agrees with the ETA-first route because the trip-success margin stays healthy. That shows the system is not forcing eco detours when they are not justified.`;
  }

  const etaDelta = recommended.metrics.scenarioDurationMin - fastest.metrics.scenarioDurationMin;
  const protectedGain = recommended.metrics.arrivalSafePct - fastest.metrics.arrivalSafePct;
  const successGain = recommended.metrics.successProbabilityPct - fastest.metrics.successProbabilityPct;
  if (confidenceMode) {
    return `${traffic.label}: the model accepts ${formatNumber(etaDelta, 1)} extra minutes on ${preset.title} to gain ${formatNumber(protectedGain, 1)} points of protected arrival battery and ${formatNumber(successGain, 0)} points of trip-success probability.`;
  }

  return `${traffic.label}: the recommendation is not the fastest route because a small ETA increase buys a meaningfully better energy outcome and lower congestion exposure.`;
}

function buildReviewerCopy(preset, fastest, recommended, confidenceMode) {
  const decision = getDecisionProfile();
  if (recommended.id === fastest.id) {
    return `An ETA-first map and our ${decision.label.toLowerCase()} mode both choose ${recommended.label}. That is still a win: the system confirms when the fastest route is already battery-safe.`;
  }

  const etaDelta = recommended.metrics.scenarioDurationMin - fastest.metrics.scenarioDurationMin;
  const energySaved = fastest.metrics.expectedKwh - recommended.metrics.expectedKwh;
  const protectedGain = recommended.metrics.arrivalSafePct - fastest.metrics.arrivalSafePct;
  const successGain = recommended.metrics.successProbabilityPct - fastest.metrics.successProbabilityPct;

  if (energySaved >= 0.05) {
    return `${preset.title}: a normal map would send the driver via ${fastest.label} in ${formatNumber(fastest.metrics.scenarioDurationMin, 1)} min. Our system chooses ${recommended.label}, accepts ${formatNumber(etaDelta, 1)} extra minutes, saves ${formatNumber(energySaved, 2)} kWh, and adds ${formatNumber(successGain, 0)} points of trip-success probability with ${formatNumber(protectedGain, 1)} more protected arrival SOC.`;
  }

  if (confidenceMode) {
    return `${preset.title}: even when expected kWh is close, the optimizer rejects ${fastest.label} because ${recommended.label} gives ${formatNumber(protectedGain, 1)} more points of protected arrival battery under uncertainty.`;
  }

  return `${preset.title}: the recommendation is an explainable battery-confidence trade-off, not just another ETA. The route wins on lower stop-go risk and stronger arrival margin.`;
}

function buildReviewerNote(preset, flipState) {
  if (!flipState.hasFlip) {
    return "Reviewer note: in this exact configuration the recommended route stays stable across the battery range, which shows the score is not oscillating unpredictably.";
  }

  if (preset.id === "airport-omr") {
    return `Reviewer note: on the OMR preset the route flips near ${formatNumber(flipState.thresholdSoc, 0)}% battery. That lets you demonstrate how the same trip changes from ETA-led to confidence-led as usable range gets tighter.`;
  }

  return `Reviewer note: on the IIT preset, keep the battery above ${formatNumber(flipState.thresholdSoc, 0)}% to show the efficient corridor. Then drag below that point to demonstrate the switch into the safer reserve approach.`;
}

function buildReasonList(recommended, fastest) {
  const scenario = getTraffic();
  const driver = getDriver();
  const decision = getDecisionProfile();
  const reasons = [];
  const energySaved = fastest.metrics.expectedKwh - recommended.metrics.expectedKwh;
  const protectedGain = recommended.metrics.arrivalSafePct - fastest.metrics.arrivalSafePct;
  const successGain = recommended.metrics.successProbabilityPct - fastest.metrics.successProbabilityPct;
  const slowShareGain = (fastest.metrics.adjustedSlowShare - recommended.metrics.adjustedSlowShare) * 100;

  if (protectedGain > 0.4) {
    reasons.push({
      title: "Higher arrival confidence",
      text: `Protected arrival improves by ${formatNumber(protectedGain, 1)} points, which is the main reason this route is safer under uncertainty.`
    });
  }

  if (energySaved > 0.05) {
    reasons.push({
      title: "Lower expected battery draw",
      text: `Compared with ${fastest.label}, this route is expected to save about ${formatNumber(energySaved, 2)} kWh.`
    });
  }

  if (slowShareGain > 1) {
    reasons.push({
      title: "Cleaner stop-go profile",
      text: `Modeled slow-traffic exposure drops by ${formatNumber(slowShareGain, 1)} points, which cuts reserve demand.`
    });
  }

  reasons.push({
    title: "Decision mode effect",
    text: `${decision.label} is active, so the model ${decision.key === "fast" ? "heavily penalizes time loss" : decision.key === "safe" ? "heavily rewards reliable arrival" : "balances ETA, battery use, and reliability"} in the final score.`
  });

  reasons.push({
    title: "Driver and scenario effect",
    text: `${driver.label} with ${scenario.label.toLowerCase()} changes the stop-go penalty, cabin load, and uncertainty reserve.`
  });

  reasons.push({
    title: "Trip success probability",
    text: `The recommendation raises modeled trip success from ${formatNumber(fastest.metrics.successProbabilityPct, 0)}% to ${formatNumber(recommended.metrics.successProbabilityPct, 0)}%, a gain of ${formatNumber(successGain, 0)} points.`
  });

  return reasons.slice(0, 4);
}

function buildSensitivityRows(preset) {
  const fast = annotateRoutes(preset, { decisionValue: 0 }).recommended;
  const balanced = annotateRoutes(preset, { decisionValue: 50 }).recommended;
  const safe = annotateRoutes(preset, { decisionValue: 100 }).recommended;

  return [
    { label: "Fastest first", key: "fast", route: fast.label },
    { label: "Balanced tradeoff", key: "balanced", route: balanced.label },
    { label: "Safest arrival", key: "safe", route: safe.label }
  ];
}

function getRiskTone(metrics) {
  if (metrics.successProbabilityPct < 70) {
    return "danger";
  }
  if (metrics.successProbabilityPct < 85) {
    return "warning";
  }
  return "safe";
}

function buildMapStats(recommended, fastest) {
  const etaDelta = recommended.metrics.scenarioDurationMin - fastest.metrics.scenarioDurationMin;
  const decision = getDecisionProfile();
  return [
    { label: "Road ETA", value: `${formatNumber(recommended.durationMin, 1)} min` },
    { label: "Traffic ETA", value: `${formatNumber(recommended.metrics.scenarioDurationMin, 1)} min` },
    { label: "Trip success", value: `${formatNumber(recommended.metrics.successProbabilityPct, 0)}%` },
    { label: "Protected arrival", value: `${formatNumber(recommended.metrics.arrivalSafePct, 1)}%` },
    { label: "Decision mode", value: decision.label },
    { label: "Vs ETA-first", value: `${signedNumber(etaDelta, 1)} min` }
  ];
}

function renderMapStats(stats) {
  refs.mapStats.innerHTML = stats.map((stat) => `
    <div class="map-stat">
      <div class="stat-label">${stat.label}</div>
      <span class="stat-value">${stat.value}</span>
    </div>
  `).join("");
}

function getMapAnnotations(preset) {
  const shared = [
    {
      label: "Airport",
      lon: preset.origin.lon,
      lat: preset.origin.lat,
      kind: "place",
      dx: 16,
      dy: -14
    }
  ];

  if (preset.id === "airport-iit") {
    return shared.concat([
      {
        label: "Guindy",
        lon: 80.212,
        lat: 13.008,
        kind: "place",
        dx: 10,
        dy: -10
      },
      {
        label: "Velachery",
        lon: 80.219,
        lat: 12.981,
        kind: "place",
        dx: 10,
        dy: 18
      },
      {
        label: "Taramani",
        lon: 80.241,
        lat: 12.985,
        kind: "place",
        dx: 12,
        dy: -12
      },
      {
        label: "Inner Ring corridor",
        lon: 80.203,
        lat: 13.006,
        kind: "corridor",
        dx: 6,
        dy: -16
      },
      {
        label: "IIT Madras",
        lon: preset.destination.lon,
        lat: preset.destination.lat,
        kind: "destination",
        dx: 14,
        dy: -14
      }
    ]);
  }

  return shared.concat([
    {
      label: "Pallavaram",
      lon: 80.176,
      lat: 12.959,
      kind: "place",
      dx: 10,
      dy: -10
    },
    {
      label: "Guindy",
      lon: 80.212,
      lat: 13.008,
      kind: "place",
      dx: 10,
      dy: -10
    },
    {
      label: "Perungudi",
      lon: 80.233,
      lat: 12.969,
      kind: "place",
      dx: 8,
      dy: -10
    },
    {
      label: "OMR / Rajiv Gandhi Salai",
      lon: 80.226,
      lat: 12.949,
      kind: "corridor",
      dx: 6,
      dy: -14
    },
    {
      label: "Thoraipakkam",
      lon: preset.destination.lon,
      lat: preset.destination.lat,
      kind: "destination",
      dx: 14,
      dy: -12
    }
  ]);
}

function renderMap(preset, annotated, selectedRoute, recommended) {
  const width = 900;
  const height = 430;
  const padding = 34;
  const points = [];

  annotated.forEach((route) => {
    route.geometry.forEach((point) => {
      points.push({ lon: point[0], lat: point[1] });
    });
  });

  points.push({ lon: preset.origin.lon, lat: preset.origin.lat });
  points.push({ lon: preset.destination.lon, lat: preset.destination.lat });

  const lonMin = Math.min(...points.map((point) => point.lon));
  const lonMax = Math.max(...points.map((point) => point.lon));
  const latMin = Math.min(...points.map((point) => point.lat));
  const latMax = Math.max(...points.map((point) => point.lat));
  const lonRange = lonMax - lonMin || 1;
  const latRange = latMax - latMin || 1;

  function project(lon, lat) {
    const x = padding + ((lon - lonMin) / lonRange) * (width - (padding * 2));
    const y = height - padding - ((lat - latMin) / latRange) * (height - (padding * 2));
    return [x, y];
  }

  const guideLines = [];
  for (let i = 1; i <= 4; i += 1) {
    const y = 38 + (i * 76);
    guideLines.push(`<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="rgba(16,54,55,0.05)" stroke-width="1"></line>`);
  }
  for (let i = 1; i <= 6; i += 1) {
    const x = 56 + (i * 122);
    guideLines.push(`<line x1="${x}" y1="0" x2="${x}" y2="${height}" stroke="rgba(16,54,55,0.05)" stroke-width="1"></line>`);
  }

  const routeLines = annotated.map((route) => {
    const path = route.geometry.map((point) => project(point[0], point[1]).join(",")).join(" ");
    const isSelected = route.id === selectedRoute.id;
    const isRecommended = route.id === recommended.id;
    return `
      <polyline
        class="route-line ${isSelected ? "selected" : ""}"
        points="${path}"
        stroke="${route.color}"
        stroke-width="${isSelected ? 8 : isRecommended ? 5 : 4}"
        opacity="${isSelected ? 0.98 : isRecommended ? 0.62 : 0.28}">
      </polyline>
    `;
  }).join("");

  const roadLayer = annotated.map((route) => {
    const path = route.geometry.map((point) => project(point[0], point[1]).join(",")).join(" ");
    return `
      <polyline
        class="route-underlay"
        points="${path}"
        stroke="rgba(16,54,55,0.10)"
        stroke-width="10"
        opacity="0.7">
      </polyline>
    `;
  }).join("");

  const [ox, oy] = project(preset.origin.lon, preset.origin.lat);
  const [dx, dy] = project(preset.destination.lon, preset.destination.lat);
  const selectedMid = selectedRoute.geometry[Math.floor(selectedRoute.geometry.length * 0.62)];
  const [lx, ly] = project(selectedMid[0], selectedMid[1]);
  const annotations = getMapAnnotations(preset).map((annotation) => {
    const [x, y] = project(annotation.lon, annotation.lat);
    return `
      <g class="map-annotation ${annotation.kind}" transform="translate(${x}, ${y})">
        <circle class="map-node-dot" r="${annotation.kind === "corridor" ? 2.6 : 3.4}"></circle>
        <text class="map-label ${annotation.kind}" x="${annotation.dx}" y="${annotation.dy}">
          ${annotation.label}
        </text>
      </g>
    `;
  }).join("");

  refs.routeMap.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
    ${guideLines.join("")}
    <path d="M725 0 C782 66 808 126 824 185 C846 270 874 334 900 430 L900 0 Z" fill="rgba(26,123,119,0.08)"></path>
    <path d="M676 24 C710 64 742 118 764 186 C784 250 810 316 860 414" fill="none" stroke="rgba(26,123,119,0.18)" stroke-width="2.2" stroke-dasharray="10 11"></path>
    <text x="38" y="34" font-family="Trebuchet MS, Lucida Sans Unicode, sans-serif" font-size="11" letter-spacing="0.18em" fill="rgba(16,54,55,0.42)">CHENNAI ROUTE STAGE</text>
    <text x="822" y="34" font-family="Trebuchet MS, Lucida Sans Unicode, sans-serif" font-size="12" font-weight="700" fill="rgba(16,54,55,0.54)">N</text>
    <line x1="826" y1="54" x2="826" y2="24" stroke="rgba(16,54,55,0.46)" stroke-width="2"></line>
    <path d="M826 17 L820 28 L832 28 Z" fill="rgba(16,54,55,0.46)"></path>
    <line x1="52" y1="388" x2="136" y2="388" stroke="rgba(16,54,55,0.52)" stroke-width="3" stroke-linecap="round"></line>
    <text x="52" y="409" font-family="Trebuchet MS, Lucida Sans Unicode, sans-serif" font-size="11" fill="rgba(16,54,55,0.46)">Approx route view</text>
    ${roadLayer}
    ${routeLines}
    ${annotations}
    <g class="marker-pulse" transform="translate(${ox}, ${oy})">
      <circle r="11" fill="white" stroke="${colors[0]}" stroke-width="3"></circle>
      <circle r="4" fill="${colors[0]}"></circle>
    </g>
    <g class="marker-pulse" transform="translate(${dx}, ${dy})">
      <circle r="11" fill="white" stroke="${colors[1]}" stroke-width="3"></circle>
      <path d="M0 -6 L4 0 L0 6 L-4 0 Z" fill="${colors[1]}"></path>
    </g>
    <text class="map-route-label" x="${lx + 12}" y="${ly - 12}" fill="${selectedRoute.color}">
      ${selectedRoute.label}
    </text>
  `;

  refs.selectedTitle.textContent = selectedRoute.label;
  refs.selectedCaption.textContent = selectedRoute.id === recommended.id
    ? `Recommended under the current battery, driver, and scenario settings. Traffic ETA rises to ${formatNumber(selectedRoute.metrics.scenarioDurationMin, 1)} min, trip success is ${formatNumber(selectedRoute.metrics.successProbabilityPct, 0)}%, and protected arrival stays at ${formatNumber(selectedRoute.metrics.arrivalSafePct, 1)}%.`
    : `You are inspecting an alternative. The current recommendation remains ${recommended.label} because its combined energy-confidence score and success probability are stronger.`;
  refs.mapLegend.innerHTML = annotated.map((route) => `
    <div class="legend-item ${route.id === selectedRoute.id ? "selected" : ""}">
      <span class="legend-swatch" style="background:${route.color};"></span>
      <span>${route.label}</span>
      ${route.id === recommended.id ? '<span class="legend-tag">Recommended</span>' : ""}
    </div>
  `).join("");
}

function renderComparison(fastest, recommended, confidenceMode) {
  const traffic = getTraffic();
  const decision = getDecisionProfile();
  const etaDelta = recommended.metrics.scenarioDurationMin - fastest.metrics.scenarioDurationMin;
  const energyDelta = recommended.metrics.expectedKwh - fastest.metrics.expectedKwh;
  const robustGain = recommended.metrics.arrivalSafePct - fastest.metrics.arrivalSafePct;
  const exposureDelta = (fastest.metrics.adjustedSlowShare - recommended.metrics.adjustedSlowShare) * 100;
  const successGain = recommended.metrics.successProbabilityPct - fastest.metrics.successProbabilityPct;

  refs.comparisonGrid.innerHTML = `
    <article class="comparison-card card fade-card" style="--delay: 80ms;">
      <div class="summary-kicker">ETA-first baseline</div>
      <div class="summary-headline">${fastest.label}</div>
      <div class="summary-detail">This is what a normal map would choose. Under ${traffic.label.toLowerCase()}, it optimizes traffic ETA only and ignores route-level battery reliability.</div>
      <div class="value-stack">
        <div class="value-row"><span>Road ETA</span><strong>${formatNumber(fastest.durationMin, 1)} min</strong></div>
        <div class="value-row"><span>Traffic ETA</span><strong>${formatNumber(fastest.metrics.scenarioDurationMin, 1)} min</strong></div>
        <div class="value-row"><span>Expected use</span><strong>${formatNumber(fastest.metrics.expectedKwh, 2)} kWh</strong></div>
        <div class="value-row"><span>Trip success</span><strong>${formatNumber(fastest.metrics.successProbabilityPct, 0)}%</strong></div>
      </div>
    </article>

    <article class="comparison-card card fade-card" style="--delay: 140ms;">
      <div class="summary-kicker">Our system</div>
      <div class="summary-headline">${recommended.label}</div>
      <div class="summary-detail">${confidenceMode ? "The score is reliability-led because route volatility matters more under the current battery and scenario." : `The score follows ${decision.label.toLowerCase()} and balances ETA, expected energy, and reliability reserve.`}</div>
      <div class="value-stack">
        <div class="value-row"><span>Road ETA</span><strong>${formatNumber(recommended.durationMin, 1)} min</strong></div>
        <div class="value-row"><span>Traffic ETA</span><strong>${formatNumber(recommended.metrics.scenarioDurationMin, 1)} min</strong></div>
        <div class="value-row"><span>Expected use</span><strong>${formatNumber(recommended.metrics.expectedKwh, 2)} kWh</strong></div>
        <div class="value-row"><span>Trip success</span><strong>${formatNumber(recommended.metrics.successProbabilityPct, 0)}%</strong></div>
      </div>
    </article>

    <article class="comparison-card card fade-card" style="--delay: 200ms;">
      <div class="summary-kicker">Measured trade-off</div>
      <div class="summary-headline">${recommended.id === fastest.id ? "No unnecessary override" : "Why the recommendation changed"}</div>
      <div class="delta-row" style="margin-top: 12px;">
        <span class="delta-chip ${deltaClass(-etaDelta, false)}">${signedNumber(etaDelta, 1)} min</span>
        <span class="delta-chip ${deltaClass(energyDelta, false)}">${signedNumber(energyDelta, 2)} kWh</span>
        <span class="delta-chip ${deltaClass(robustGain, true)}">${signedNumber(robustGain, 1)}% protected SOC</span>
        <span class="delta-chip ${deltaClass(successGain, true)}">${signedNumber(successGain, 0)} pts success</span>
        <span class="delta-chip ${deltaClass(exposureDelta, true)}">${signedNumber(exposureDelta, 1)} pts slow share</span>
      </div>
      <div class="summary-detail" style="margin-top: 12px;">
        ${recommended.id === fastest.id
          ? "The route already wins on time and stays battery-safe, so the optimizer does not invent an eco detour."
          : "This is the core contrast for the reviewer: a small ETA increase can be rational when it reduces energy risk and strengthens the arrival buffer."}
      </div>
    </article>

    <article class="comparison-card card fade-card" style="--delay: 260ms;">
      <div class="summary-kicker">Why significant</div>
      <div class="summary-headline">${confidenceMode ? "Protected arrival matters" : "Battery-aware route choice"}</div>
      <div class="summary-detail">
        ${recommended.id === fastest.id
          ? "The system validates the normal route when it is already robust enough. That makes the model practical, not dogmatic."
          : confidenceMode
            ? "An EV driver cares about the arrival buffer that survives congestion uncertainty, not just a single nominal ETA."
            : "This turns routing into decision support: the route is explainable in terms of speed profile, stop-go load, slope, and battery outcome."}
      </div>
    </article>
  `;
}

function renderFlipState(preset, flipState, recommended) {
  refs.flipCurrent.style.left = `${clamp(state.soc, 5, 95)}%`;

  if (!flipState.hasFlip) {
    refs.flipCard.classList.remove("flip-active");
    refs.flipZone.style.width = "0%";
    refs.flipThreshold.style.left = "100%";
    refs.flipHeadline.textContent = "Recommendation stays stable";
    refs.flipCopy.textContent = `${recommended.label} remains the best-scoring route across the tested battery band for this preset.`;
    return;
  }

  refs.flipZone.style.width = `${clamp(flipState.thresholdSoc, 5, 95)}%`;
  refs.flipThreshold.style.left = `${clamp(flipState.thresholdSoc, 5, 95)}%`;

  if (state.soc <= flipState.thresholdSoc) {
    refs.flipHeadline.textContent = `${flipState.lowRoute.label} is active below ${formatNumber(flipState.thresholdSoc, 0)}%`;
    refs.flipCopy.textContent = `${preset.title}: above ${formatNumber(flipState.thresholdSoc, 0)}% the score prefers ${flipState.highRoute.label}. Below that threshold it flips to ${flipState.lowRoute.label} to protect arrival confidence. You are currently at ${state.soc}%.`;
  } else {
    refs.flipHeadline.textContent = `Recommendation flips below ${formatNumber(flipState.thresholdSoc, 0)}%`;
    refs.flipCopy.textContent = `${preset.title}: ${flipState.highRoute.label} currently leads. Drop below ${formatNumber(flipState.thresholdSoc, 0)}% battery and the optimizer switches to ${flipState.lowRoute.label}.`;
  }
}

function renderEnergyGrid(selectedRoute, annotated, recommended) {
  const traffic = getTraffic();
  const components = [
    { key: "baseDriveKwh", label: "Base drive", className: "base" },
    { key: "congestionPenaltyKwh", label: "Congestion penalty", className: "congestion" },
    { key: "speedPenaltyKwh", label: "Cruise-speed penalty", className: "speed" },
    { key: "climbKwh", label: "Climb cost", className: "climb" },
    { key: "cabinKwh", label: "Cabin load", className: "cabin" },
    { key: "reserveKwh", label: "Confidence reserve", className: "reserve", source: "reserve" }
  ];

  const selectedParts = components.map((item) => {
    const value = item.source === "reserve"
      ? selectedRoute.metrics.reserveKwh
      : selectedRoute.metrics.components[item.key];
    return { ...item, value };
  });
  const totalPositive = selectedParts.reduce((sum, part) => sum + part.value, 0) || 1;
  const maxRobust = Math.max(...annotated.map((route) => route.metrics.robustKwh), selectedRoute.metrics.robustKwh);
  const creditWidth = clamp((selectedRoute.metrics.components.regenCreditKwh / totalPositive) * 100, 4, 42);

  refs.energyGrid.innerHTML = `
    <article class="energy-card card fade-card" style="--delay: 120ms;">
      <div class="summary-kicker">Energy anatomy</div>
      <div class="summary-headline">${selectedRoute.label}</div>
      <div class="energy-stack">
        <div class="energy-total">${formatNumber(selectedRoute.metrics.robustKwh, 2)} kWh robust need</div>
        <div class="summary-detail">Road ETA is ${formatNumber(selectedRoute.durationMin, 1)} min. Under ${traffic.label.toLowerCase()}, modeled trip time becomes ${formatNumber(selectedRoute.metrics.scenarioDurationMin, 1)} min. The reserve is the battery protection layer on top of expected use.</div>
        <div class="energy-bar">
          ${selectedParts.map((part) => `
            <div
              class="energy-segment ${part.className}"
              style="width: ${(part.value / totalPositive) * 100}%"
              title="${part.label}: ${formatNumber(part.value, 2)} kWh">
            </div>
          `).join("")}
        </div>
        <div class="energy-credit">
          <span class="detail-text">Regen credit</span>
          <div class="energy-credit-bar" style="width: ${creditWidth}%"></div>
          <strong>-${formatNumber(selectedRoute.metrics.components.regenCreditKwh, 2)} kWh</strong>
        </div>
        <div class="energy-list">
          ${selectedParts.map((part) => `
            <div class="energy-row">
              <div class="energy-row-head">
                <span>${part.label}</span>
                <strong>${formatNumber(part.value, 2)} kWh</strong>
              </div>
            </div>
          `).join("")}
        </div>
      </div>
    </article>

    <article class="energy-card card fade-card" style="--delay: 180ms;">
      <div class="summary-kicker">Route contrast</div>
      <div class="summary-headline">Why the ranking changes</div>
      <div class="summary-detail">A normal route engine compares ETA. Our policy score compares expected use, reserve, and time penalty. The tracks below show expected use first, then the uncertainty reserve layered on top.</div>
      <div class="energy-route-list">
        ${annotated.map((route) => {
          const expectedWidth = (route.metrics.expectedKwh / maxRobust) * 100;
          const robustWidth = (route.metrics.robustKwh / maxRobust) * 100;
          return `
            <div class="energy-route-row ${route.id === recommended.id ? "recommended" : ""}">
              <div class="energy-route-head">
                <span>${route.label}</span>
                <strong>${formatNumber(route.metrics.scenarioDurationMin, 1)} min | ${formatNumber(route.metrics.arrivalSafePct, 1)}%</strong>
              </div>
              <div class="energy-route-track">
                <div class="energy-route-fill" style="width: ${expectedWidth}%; background: ${route.color};"></div>
                <div class="energy-route-reserve" style="width: ${robustWidth}%;"></div>
              </div>
              <div class="energy-row-head">
                <span>Expected ${formatNumber(route.metrics.expectedKwh, 2)} kWh + reserve ${formatNumber(route.metrics.reserveKwh, 2)} kWh</span>
                <strong>${route.id === recommended.id ? "Recommended" : formatNumber(route.policyScore, 2)}</strong>
              </div>
            </div>
          `;
        }).join("")}
      </div>
    </article>
  `;
}

function renderRiskGrid(preset, recommended, fastest, flipState) {
  const tone = getRiskTone(recommended.metrics);
  const decision = getDecisionProfile();
  const driver = getDriver();
  const scenario = getTraffic();
  const successDelta = recommended.metrics.successProbabilityPct - fastest.metrics.successProbabilityPct;
  const bandLow = formatNumber(recommended.metrics.riskBandLowPct, 1);
  const bandHigh = formatNumber(recommended.metrics.riskBandHighPct, 1);

  refs.riskGrid.innerHTML = `
    <article class="risk-card card fade-card" style="--delay: 120ms;">
      <div class="summary-kicker">Reliability view</div>
      <div class="summary-headline">${recommended.metrics.riskLevel}</div>
      <div class="risk-shell">
        <div class="risk-ring ${tone === "danger" ? "danger" : tone === "warning" ? "warning" : ""}" style="--pct: ${recommended.metrics.successProbabilityPct}%;">
          <div class="risk-ring-inner">
            <span class="stat-label">Trip success</span>
            <div class="risk-ring-value">${formatNumber(recommended.metrics.successProbabilityPct, 0)}%</div>
            <span class="detail-text">reliable arrival</span>
          </div>
        </div>
        <div class="risk-copy">
          <p class="summary-detail">Instead of only showing expected arrival battery, the system estimates whether this trip remains feasible under stressed conditions. The likely arrival band is <strong>${bandLow}% to ${bandHigh}%</strong>.</p>
          <div class="risk-badge-row">
            <span class="risk-badge ${recommended.metrics.riskLevel === "Low risk" ? "low" : recommended.metrics.riskLevel === "Medium risk" ? "medium" : "high"}">${recommended.metrics.riskLevel}</span>
            <span class="risk-badge ${successDelta >= 0 ? "low" : "high"}">${signedNumber(successDelta, 0)} pts vs ETA-first</span>
          </div>
        </div>
      </div>
      <div class="risk-metric-grid">
        <div class="risk-metric">
          <span class="stat-label">Nominal arrival</span>
          <strong>${formatNumber(recommended.metrics.arrivalPct, 1)}%</strong>
        </div>
        <div class="risk-metric">
          <span class="stat-label">Protected arrival</span>
          <strong>${formatNumber(recommended.metrics.arrivalSafePct, 1)}%</strong>
        </div>
        <div class="risk-metric">
          <span class="stat-label">Confidence reserve</span>
          <strong>${formatNumber(recommended.metrics.reserveKwh, 2)} kWh</strong>
        </div>
      </div>
    </article>

    <article class="risk-card card fade-card" style="--delay: 180ms;">
      <div class="summary-kicker">Parameter sensitivity</div>
      <div class="summary-headline">What is driving the answer</div>
      <div class="summary-detail">${scenario.note} ${driver.note} ${decision.note}</div>
      <div class="mode-row">
        ${buildSensitivityRows(preset).map((row) => `
          <span class="mode-pill ${row.route === recommended.label ? `active ${row.key}` : ""}">
            <span>${row.label}</span>
            <strong>${row.route}</strong>
          </span>
        `).join("")}
      </div>
      <div class="risk-badge-row">
        <span class="reason-pill">Scenario: ${scenario.label}</span>
        <span class="reason-pill">Driver: ${driver.label}</span>
        <span class="reason-pill">Preference: ${decision.label}</span>
      </div>
      <div class="summary-detail">
        ${flipState.hasFlip
          ? `If the battery drops below about ${formatNumber(flipState.thresholdSoc, 0)}%, the route choice flips from ${flipState.highRoute.label} to ${flipState.lowRoute.label}.`
          : "Under the current profile, the recommendation is stable across the battery band, which is useful evidence that the answer is not brittle."}
      </div>
    </article>
  `;
}

function renderRationaleGrid(preset, recommended, fastest) {
  const reasons = buildReasonList(recommended, fastest);
  const decision = getDecisionProfile();
  const traffic = getTraffic();
  const driver = getDriver();

  refs.rationaleGrid.innerHTML = `
    <article class="rationale-card card fade-card" style="--delay: 130ms;">
      <div class="summary-kicker">Why this route won</div>
      <div class="summary-headline">${recommended.label}</div>
      <div class="reason-list">
        ${reasons.map((reason) => `
          <div class="reason-item">
            <strong>${reason.title}</strong>
            <div class="detail-text">${reason.text}</div>
          </div>
        `).join("")}
      </div>
    </article>

    <article class="rationale-card card fade-card" style="--delay: 190ms;">
      <div class="summary-kicker">Reviewer framing</div>
      <div class="summary-headline">How to explain the innovation</div>
      <div class="reason-list">
        <div class="reason-item">
          <strong>Not another ETA app</strong>
          <div class="detail-text">A normal maps app would stop at travel time. This system adds a battery-success layer on top of real routes.</div>
        </div>
        <div class="reason-item">
          <strong>Research idea translated into product UX</strong>
          <div class="detail-text">Robust EV-routing research talks about uncertainty and reliability. Here, that becomes trip success probability, protected arrival SOC, and a visible flip threshold.</div>
        </div>
        <div class="reason-item">
          <strong>Live controls prove sensitivity</strong>
          <div class="detail-text">Switching between ${decision.label.toLowerCase()}, ${driver.label.toLowerCase()}, and ${traffic.label.toLowerCase()} conditions changes the recommendation and explains why.</div>
        </div>
        <div class="reason-item">
          <strong>Presentation line</strong>
          <div class="detail-text">${preset.title}: the system is optimizing for reliable EV arrival under uncertainty, not just shortest time.</div>
        </div>
      </div>
    </article>
  `;
}

function renderSummary(preset, recommended, fastest, confidenceMode) {
  const etaDelta = recommended.metrics.scenarioDurationMin - fastest.metrics.scenarioDurationMin;
  const energySaved = fastest.metrics.expectedKwh - recommended.metrics.expectedKwh;
  const protectedGain = recommended.metrics.arrivalSafePct - fastest.metrics.arrivalSafePct;
  const decision = getDecisionProfile();
  const protectedVerdict = recommended.metrics.successProbabilityPct < 70
    ? "Arrival reliability is thin"
    : recommended.metrics.successProbabilityPct < 85
      ? "Arrival reliability is workable"
      : "Arrival reliability is strong";

  refs.summaryGrid.innerHTML = `
    <article class="summary-card card fade-card" style="--delay: 110ms;">
      <div class="summary-kicker">Recommendation</div>
      <div class="summary-headline">${recommended.label}</div>
      <div class="summary-detail">
        ${confidenceMode
          ? "Chosen because the route preserves a stronger arrival buffer and a better trip-success probability under the current uncertainty profile."
          : "Chosen because it produces the best combined score across expected energy, reliability reserve, risk, and ETA penalty."}
      </div>
    </article>

    <article class="summary-card card fade-card" style="--delay: 170ms;">
      <div class="summary-kicker">Decision score</div>
      <div class="big-figure">${formatNumber(recommended.metrics.expectedKwh, 2)} x ${formatNumber(decision.energyWeight, 2)} + ${formatNumber(recommended.metrics.reserveKwh, 2)} x ${formatNumber(getReserveWeight() * decision.reserveScale, 2)}</div>
      <div class="summary-detail">The active mode is ${decision.label.toLowerCase()}. That changes how strongly the optimizer prices ETA loss versus reliability reserve. Final score: ${formatNumber(recommended.policyScore, 2)}.</div>
    </article>

    <article class="summary-card card fade-card" style="--delay: 230ms;">
      <div class="summary-kicker">Reliability verdict</div>
      <div class="summary-headline">${protectedVerdict}</div>
      <div class="summary-detail">
        ${recommended.id === fastest.id
          ? `${preset.title}: the fastest route also remains reliable, so the system confirms it instead of overriding it.`
          : `${preset.title}: spending ${formatNumber(etaDelta, 1)} extra minutes buys ${formatNumber(Math.max(energySaved, 0), 2)} kWh lower expected use, ${formatNumber(protectedGain, 1)} more protected arrival SOC, and a materially stronger success probability.`}
      </div>
    </article>
  `;
}

function renderRouteCards(annotated, recommended) {
  refs.routesGrid.innerHTML = annotated.map((route, index) => {
    const isSelected = route.id === state.selectedRouteId;
    return `
      <button class="route-card card fade-card ${isSelected ? "active" : ""}" style="--delay: ${120 + (index * 70)}ms;" data-route-id="${route.id}">
        <div class="route-head">
          <div>
            <div class="route-title">${route.label}</div>
            <div class="route-copy">${route.description || ""}</div>
          </div>
          <div class="badge-row">
            ${route.badges.map((badge) => `<span class="badge ${badge.type}">${badge.label}</span>`).join("")}
            ${route.id === recommended.id ? '<span class="badge energy">Recommended</span>' : ""}
            ${isSelected ? '<span class="badge selected">Selected</span>' : ""}
          </div>
        </div>

        <div class="route-metrics">
          <div class="metric-card">
            <div class="metric-label">Road ETA</div>
            <span class="metric-value">${formatNumber(route.durationMin, 1)} min</span>
          </div>
          <div class="metric-card">
            <div class="metric-label">Traffic ETA</div>
            <span class="metric-value">${formatNumber(route.metrics.scenarioDurationMin, 1)} min</span>
          </div>
          <div class="metric-card">
            <div class="metric-label">Distance</div>
            <span class="metric-value">${formatNumber(route.distanceKm, 2)} km</span>
          </div>
          <div class="metric-card">
            <div class="metric-label">Expected use</div>
            <span class="metric-value">${formatNumber(route.metrics.expectedKwh, 2)} kWh</span>
          </div>
          <div class="metric-card">
            <div class="metric-label">Reserve</div>
            <span class="metric-value">${formatNumber(route.metrics.reserveKwh, 2)} kWh</span>
          </div>
          <div class="metric-card">
            <div class="metric-label">Trip success</div>
            <span class="metric-value">${formatNumber(route.metrics.successProbabilityPct, 0)}%</span>
          </div>
        </div>

        <div class="route-risk-row">
          <span class="risk-badge ${route.metrics.riskLevel === "Low risk" ? "low" : route.metrics.riskLevel === "Medium risk" ? "medium" : "high"}">${route.metrics.riskLevel}</span>
          <span class="reason-pill">Protected ${formatNumber(route.metrics.arrivalSafePct, 1)}%</span>
          <span class="reason-pill">Band ${formatNumber(route.metrics.riskBandLowPct, 1)}% to ${formatNumber(route.metrics.riskBandHighPct, 1)}%</span>
        </div>

        <div class="battery-wrap">
          <div class="range-row">
            <span>Nominal arrival battery</span>
            <span class="range-value" style="color: ${route.color};">${formatNumber(route.metrics.arrivalPct, 1)}%</span>
          </div>
          <div class="battery-bar">
            <div class="battery-fill" style="width: ${clamp(route.metrics.arrivalPct, 0, 100)}%;"></div>
          </div>
        </div>
      </button>
    `;
  }).join("");

  refs.routesGrid.querySelectorAll("[data-route-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedRouteId = button.dataset.routeId;
      render();
    });
  });
}

function findFlipState(preset) {
  const highState = annotateRoutes(preset, { soc: 95 }).recommended;
  let thresholdSoc = null;
  let lowRoute = highState;

  for (let soc = 94; soc >= 5; soc -= 1) {
    const recommendation = annotateRoutes(preset, { soc }).recommended;
    if (recommendation.id !== highState.id) {
      thresholdSoc = soc;
      lowRoute = recommendation;
      break;
    }
  }

  return {
    hasFlip: thresholdSoc !== null,
    thresholdSoc,
    highRoute: highState,
    lowRoute
  };
}

function triggerFlipPulse(nextRecommendedId) {
  if (!state.lastRecommendedId) {
    state.lastRecommendedId = nextRecommendedId;
    return;
  }

  if (state.lastRecommendedId !== nextRecommendedId) {
    refs.flipCard.classList.add("flip-active");
    window.clearTimeout(state.flipPulseTimer);
    state.flipPulseTimer = window.setTimeout(() => {
      refs.flipCard.classList.remove("flip-active");
    }, 1100);
  }

  state.lastRecommendedId = nextRecommendedId;
}

function render() {
  const preset = getPreset();
  const traffic = getTraffic();
  const driver = getDriver();
  const decision = getDecisionProfile();
  const { annotated, recommended, fastest } = annotateRoutes(preset);
  const selectedRoute = annotated.find((route) => route.id === state.selectedRouteId) || recommended;
  const confidenceMode = isConfidenceMode(annotated);
  const flipState = findFlipState(preset);

  state.selectedRouteId = selectedRoute.id;

  refs.socValue.textContent = `${state.soc}%`;
  refs.decisionValue.textContent = decision.label;
  refs.decisionCopy.textContent = decision.note;
  refs.generatedPill.textContent = `Data cached ${data.generatedAt}`;
  refs.statusCopy.textContent = buildStatusCopy(preset, recommended, fastest, confidenceMode);
  refs.reviewerCopy.textContent = buildReviewerCopy(preset, fastest, recommended, confidenceMode);
  refs.mapTitle.textContent = preset.title;
  refs.mapSubtitle.textContent = `${preset.origin.name} to ${preset.destination.name}. Road ETA and geometry are cached from Chennai routing data; ${traffic.label.toLowerCase()} and ${driver.label.toLowerCase()} then reshape trip time, battery draw, and route risk.`;
  refs.formulaBreakdown.textContent = `${recommended.label}: road ETA ${formatNumber(recommended.durationMin, 1)} min -> modeled ETA ${formatNumber(recommended.metrics.scenarioDurationMin, 1)} min. Expected ${formatNumber(recommended.metrics.expectedKwh, 2)} kWh + reserve ${formatNumber(recommended.metrics.reserveKwh, 2)} kWh + ETA penalty ${formatNumber(recommended.etaPenaltyKwh * decision.etaWeight, 2)} + risk penalty ${formatNumber(recommended.riskPenalty, 2)} = score ${formatNumber(recommended.policyScore, 2)}.`;
  refs.reviewerNote.textContent = buildReviewerNote(preset, flipState);
  refs.creditLine.textContent = `Cached data sources: ${data.sourceNotes.routing} and ${data.sourceNotes.elevation}. This MVP adds a research-inspired robust routing layer on top of real Chennai route geometry: expected energy, trip-success probability, and an uncertainty reserve instead of shortest path alone.`;

  triggerFlipPulse(recommended.id);
  renderMapStats(buildMapStats(recommended, fastest));
  renderMap(preset, annotated, selectedRoute, recommended);
  renderComparison(fastest, recommended, confidenceMode);
  renderFlipState(preset, flipState, recommended);
  renderRiskGrid(preset, recommended, fastest, flipState);
  renderRationaleGrid(preset, recommended, fastest);
  renderEnergyGrid(selectedRoute, annotated, recommended);
  renderSummary(preset, recommended, fastest, confidenceMode);
  renderRouteCards(annotated, recommended);
}

function initSelect(select, items, labelFn) {
  select.innerHTML = items.map((item) => `<option value="${item.id}">${labelFn(item)}</option>`).join("");
}

function init() {
  initSelect(refs.tripSelect, data.presets, (preset) => preset.title);
  initSelect(refs.vehicleSelect, Object.values(vehicleProfiles), (vehicle) => `${vehicle.label} - ${vehicle.batteryKwh} kWh`);
  initSelect(refs.driverSelect, Object.values(driverProfiles), (driver) => driver.label);
  initSelect(refs.trafficSelect, Object.values(trafficProfiles), (traffic) => traffic.label);
  initSelect(refs.cabinSelect, Object.values(cabinProfiles), (cabin) => `${cabin.label} - ${formatNumber(cabin.loadKw, 2)} kW`);

  refs.tripSelect.value = state.presetId;
  refs.vehicleSelect.value = state.vehicleId;
  refs.driverSelect.value = state.driverId;
  refs.trafficSelect.value = state.trafficId;
  refs.cabinSelect.value = state.cabinId;
  refs.decisionRange.value = String(state.decisionValue);
  refs.socRange.value = String(state.soc);

  refs.tripSelect.addEventListener("change", (event) => {
    state.presetId = event.target.value;
    state.selectedRouteId = null;
    render();
  });

  refs.vehicleSelect.addEventListener("change", (event) => {
    state.vehicleId = event.target.value;
    render();
  });

  refs.driverSelect.addEventListener("change", (event) => {
    state.driverId = event.target.value;
    render();
  });

  refs.trafficSelect.addEventListener("change", (event) => {
    state.trafficId = event.target.value;
    render();
  });

  refs.cabinSelect.addEventListener("change", (event) => {
    state.cabinId = event.target.value;
    render();
  });

  refs.socRange.addEventListener("input", (event) => {
    state.soc = Number(event.target.value);
    render();
  });

  refs.decisionRange.addEventListener("input", (event) => {
    state.decisionValue = Number(event.target.value);
    render();
  });

  render();
}

init();
