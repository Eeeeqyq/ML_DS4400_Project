"use client";

import { useEffect, useState, useCallback } from "react";
import { ModelData, SchoolInput, predict } from "@/lib/predict";
import {
  PRESETS,
  FIELD_LABELS,
  CONTROL_OPTIONS,
  PREDDEG_OPTIONS,
} from "@/lib/presets";

function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  format,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
}) {
  const display = format ? format(value) : value.toFixed(2);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-sm">
        <span className="text-zinc-400">{label}</span>
        <span className="font-mono text-white">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-blue-500 h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer"
      />
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-sm text-zinc-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-zinc-800 border border-zinc-600 text-white rounded-lg px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function PayoffDisplay({ years }: { years: number | null }) {
  if (years === null) return null;

  const capped = years >= 24.5;
  const color = years < 10
    ? "text-emerald-400"
    : years < 18
      ? "text-amber-400"
      : "text-red-400";

  return (
    <div className="flex flex-col items-center gap-2 py-8">
      <div className="text-sm uppercase tracking-widest text-zinc-500">
        Estimated Payoff Time
      </div>
      <div className={`text-7xl font-bold tabular-nums ${color}`}>
        {capped ? "25+" : years.toFixed(1)}
      </div>
      <div className="text-2xl text-zinc-400">years</div>
      {capped && (
        <div className="text-sm text-red-400/80 mt-2 max-w-xs text-center">
          This school&apos;s typical graduate would likely need federal loan
          forgiveness (25-year IDR cap)
        </div>
      )}
      {!capped && years < 8 && (
        <div className="text-sm text-emerald-400/80 mt-2 max-w-xs text-center">
          Graduates from schools like this typically pay off debt relatively
          quickly
        </div>
      )}

      {/* Visual bar */}
      <div className="w-full max-w-md mt-4">
        <div className="h-3 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${
              years < 10
                ? "bg-emerald-500"
                : years < 18
                  ? "bg-amber-500"
                  : "bg-red-500"
            }`}
            style={{ width: `${Math.min(100, (years / 25) * 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-zinc-600 mt-1">
          <span>0 yr</span>
          <span>25 yr (forgiveness)</span>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [model, setModel] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activePreset, setActivePreset] = useState<number>(0);
  const [input, setInput] = useState<SchoolInput>(PRESETS[0].input);
  const [prediction, setPrediction] = useState<number | null>(null);

  useEffect(() => {
    fetch("/model.json")
      .then((r) => r.json())
      .then((data: ModelData) => {
        setModel(data);
        setLoading(false);
      });
  }, []);

  const runPrediction = useCallback(
    (inp: SchoolInput) => {
      if (!model) return;
      const result = predict(model, inp);
      setPrediction(result);
    },
    [model]
  );

  useEffect(() => {
    runPrediction(input);
  }, [input, runPrediction]);

  const applyPreset = (idx: number) => {
    setActivePreset(idx);
    setInput({ ...PRESETS[idx].input });
  };

  const updateField = (field: keyof SchoolInput, value: number | string) => {
    setActivePreset(-1);
    setInput((prev) => ({ ...prev, [field]: value }));
  };

  const enrollment = Math.round(Math.exp(input.UGDS_log ?? 0) - 1);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <div className="text-zinc-400 text-lg animate-pulse">
          Loading model (100 trees)...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      <div className="max-w-3xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold tracking-tight">
            College Debt Payoff Predictor
          </h1>
          <p className="text-zinc-400 mt-2 max-w-lg mx-auto">
            How long would it take a typical graduate to pay off their student
            loans? Powered by a Random Forest trained on 4,500 U.S. institutions
            from the College Scorecard.
          </p>
        </div>

        {/* Presets */}
        <div className="grid grid-cols-3 gap-3 mb-8">
          {PRESETS.map((preset, i) => (
            <button
              key={i}
              onClick={() => applyPreset(i)}
              className={`rounded-xl border p-4 text-left transition-all ${
                activePreset === i
                  ? "border-blue-500 bg-blue-500/10"
                  : "border-zinc-700 bg-zinc-900 hover:border-zinc-500"
              }`}
            >
              <div className="text-2xl mb-1">{preset.emoji}</div>
              <div className="font-semibold text-sm">{preset.name}</div>
              <div className="text-xs text-zinc-500 mt-1">
                {preset.description}
              </div>
            </button>
          ))}
        </div>

        {/* Prediction output */}
        <div className="bg-zinc-900 rounded-2xl border border-zinc-800 p-6 mb-8">
          <PayoffDisplay years={prediction} />
        </div>

        {/* Input controls */}
        <div className="bg-zinc-900 rounded-2xl border border-zinc-800 p-6">
          <h2 className="text-lg font-semibold mb-6">
            Adjust School Characteristics
          </h2>

          <div className="grid grid-cols-2 gap-x-8 gap-y-5">
            {/* Dropdowns */}
            <Select
              label="Institution Type"
              value={input.CONTROL}
              onChange={(v) => {
                updateField("CONTROL", v);
                if (v === "3") {
                  updateField("ADM_RATE_missing", 1);
                }
              }}
              options={CONTROL_OPTIONS}
            />
            <Select
              label="Predominant Degree"
              value={input.PREDDEG}
              onChange={(v) => updateField("PREDDEG", v)}
              options={PREDDEG_OPTIONS}
            />

            {/* Key sliders */}
            <Slider
              label={FIELD_LABELS.TUITIONFEE_OUT}
              value={input.TUITIONFEE_OUT ?? 20000}
              onChange={(v) => updateField("TUITIONFEE_OUT", v)}
              min={0}
              max={75000}
              step={500}
              format={(v) => `$${v.toLocaleString()}`}
            />
            <Slider
              label={FIELD_LABELS.NPT4}
              value={input.NPT4 ?? 15000}
              onChange={(v) => updateField("NPT4", v)}
              min={0}
              max={65000}
              step={500}
              format={(v) => `$${v.toLocaleString()}`}
            />
            <Slider
              label={FIELD_LABELS.UGDS_log}
              value={enrollment}
              onChange={(v) => updateField("UGDS_log", Math.log1p(v))}
              min={50}
              max={50000}
              step={50}
              format={(v) => v.toLocaleString()}
            />
            <Slider
              label={FIELD_LABELS.ADM_RATE}
              value={input.ADM_RATE ?? 0.5}
              onChange={(v) => {
                updateField("ADM_RATE", v);
                updateField("ADM_RATE_missing", 0);
              }}
              min={0}
              max={1}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Slider
              label={FIELD_LABELS.PCTPELL}
              value={input.PCTPELL ?? 0.4}
              onChange={(v) => updateField("PCTPELL", v)}
              min={0}
              max={1}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Slider
              label={FIELD_LABELS.PCTFLOAN}
              value={input.PCTFLOAN ?? 0.4}
              onChange={(v) => updateField("PCTFLOAN", v)}
              min={0}
              max={1}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Slider
              label={FIELD_LABELS.high_earning_share}
              value={input.high_earning_share ?? 0.15}
              onChange={(v) => updateField("high_earning_share", v)}
              min={0}
              max={1}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-xs text-zinc-600 mt-8">
          DS4400 Final Project &middot; Random Forest model (100 trees) trained
          on U.S. Dept. of Education College Scorecard data &middot; Predictions
          are estimates, not financial advice
        </p>
      </div>
    </div>
  );
}
