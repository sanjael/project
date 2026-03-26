import { Brain, AlertTriangle, CheckCircle, Info, Microscope, FileText,
         Eye, Download, ShieldAlert, Activity, BarChart3, Zap, Target, GitCompare, Crosshair } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import toast from 'react-hot-toast'
import ConfidenceBar from './ConfidenceBar'
import RiskBadge from './RiskBadge'
import { predictionService } from '../services/api'

const TUMOR_META = {
  glioma:      { label: 'Glioma',           color: 'text-red-600',    bg: 'bg-red-50',    border: 'border-red-100'    },
  meningioma:  { label: 'Meningioma',        color: 'text-orange-600', bg: 'bg-orange-50', border: 'border-orange-100' },
  pituitary:   { label: 'Pituitary Adenoma', color: 'text-yellow-600', bg: 'bg-yellow-50', border: 'border-yellow-100' },
}

const TUMOR_BAR_COLORS = {
  glioma:     '#ef4444',
  meningioma: '#f97316',
  pituitary:  '#eab308',
}

function ClassProbBar({ label, prob, maxProb }) {
  const pct = maxProb > 0 ? (prob / maxProb) * 100 : 0
  const color = TUMOR_BAR_COLORS[label] || '#6366f1'
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-500 capitalize w-20 shrink-0">{label.replace('_', ' ')}</span>
      <div className="flex-1 h-2.5 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-bold text-slate-600 w-12 text-right shrink-0">
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )
}

function UncertaintyGauge({ value, max = 0.3, label }) {
  const pct = Math.min((value / max) * 100, 100)
  const isHigh = value > 0.1
  return (
    <div>
      <div className="flex justify-between items-end mb-1">
        <span className="text-xs font-medium text-slate-500">{label}</span>
        <span className={`text-xs font-mono font-bold ${isHigh ? 'text-amber-600' : 'text-emerald-600'}`}>
          ±{value.toFixed(4)}
        </span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8 }}
          className={`h-full rounded-full ${isHigh ? 'bg-amber-400' : 'bg-emerald-400'}`}
        />
      </div>
    </div>
  )
}

function BboxCanvas({ localization }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !localization?.bbox) return
    const ctx = canvas.getContext('2d')
    const [nx, ny, nw, nh] = localization.bbox
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)
    ctx.strokeStyle = '#facc15'
    ctx.lineWidth = 2
    ctx.setLineDash([5, 3])
    ctx.strokeRect(nx * W, ny * H, nw * W, nh * H)
    ctx.setLineDash([])
    ctx.fillStyle = 'rgba(250,204,21,0.15)'
    ctx.fillRect(nx * W, ny * H, nw * W, nh * H)
  }, [localization])
  return <canvas ref={canvasRef} width={224} height={224} className="absolute inset-0 w-full h-full" />
}

export default function ResultsPanel({ result }) {
  const [downloading, setDownloading] = useState(false)
  const [heatmapMode, setHeatmapMode] = useState('gradcam')  // 'gradcam' | 'scorecam' | 'comparison'
  const [opacity, setOpacity] = useState(0.7)
  const [showBbox, setShowBbox] = useState(true)

  const {
    tumor_detected, tumor_type, confidence, uncertainty, entropy, reliability,
    risk_level, clinical_note, recommendation, heatmap_image, scorecam_image,
    comparison_strip, all_class_probs, tta_agreement, localization
  } = result

  const tumorMeta = tumor_type ? TUMOR_META[tumor_type] : null
  const maxProb = all_class_probs ? Math.max(...Object.values(all_class_probs)) : 1
  const entropyPct = Math.round((entropy || 0) * 100)
  const ttaAgreementPct = Math.round((tta_agreement || 1) * 100)

  const currentHeatmap = heatmapMode === 'gradcam'
    ? heatmap_image
    : heatmapMode === 'scorecam'
      ? (scorecam_image || heatmap_image)
      : (comparison_strip || heatmap_image)

  const handleDownloadReport = async () => {
    setDownloading(true)
    try {
      const blob = await predictionService.getReport(result)
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `NeuroScan_Report_${tumor_type || 'NoTumor'}.pdf`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      toast.success('Report downloaded successfully')
    } catch {
      toast.error('Failed to generate report')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Download Action */}
      <div className="flex justify-end">
        <button
          onClick={handleDownloadReport}
          disabled={downloading}
          className="btn-secondary py-2 text-xs transition-all hover:ring-2 hover:ring-neural-100"
        >
          {downloading ? 'Generating...' : <><Download className="w-3.5 h-3.5" /> Download Clinical Report</>}
        </button>
      </div>

      {/* Detection status banner */}
      <div className={`flex items-center gap-3 p-4 rounded-2xl border ${
        tumor_detected ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
      }`}>
        {tumor_detected
          ? <AlertTriangle className="w-6 h-6 text-red-600 shrink-0" />
          : <CheckCircle className="w-6 h-6 text-green-600 shrink-0" />
        }
        <div>
          <p className={`font-semibold text-sm ${tumor_detected ? 'text-red-800' : 'text-green-800'}`}>
            {tumor_detected ? '⚠️ Tumor Detected' : '✅ No Tumor Detected'}
          </p>
          <p className={`text-xs mt-0.5 ${tumor_detected ? 'text-red-600' : 'text-green-600'}`}>
            {reliability}
          </p>
        </div>
      </div>

      {/* Tumor type */}
      {tumor_detected && tumorMeta && (
        <div className={`p-4 rounded-2xl border ${tumorMeta.bg} ${tumorMeta.border}`}>
          <div className="flex items-center gap-2 mb-1">
            <Microscope className={`w-4 h-4 ${tumorMeta.color}`} />
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Tumor Type</span>
          </div>
          <p className={`font-display text-2xl ${tumorMeta.color}`}>{tumorMeta.label}</p>
        </div>
      )}

      {/* Confidence + Metrics Grid */}
      <div className="card space-y-4">
        {/* Main confidence */}
        <div>
          <div className="flex justify-between items-end mb-1.5">
            <span className="text-xs font-semibold text-slate-500 uppercase">Detection Confidence</span>
            <span className="text-lg font-display text-neural-600">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <ConfidenceBar value={confidence} />
        </div>

        {/* Metrics row */}
        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-50">
          <UncertaintyGauge value={uncertainty} max={0.3} label="MC Dropout Uncertainty" />
          <div>
            <div className="flex justify-between items-end mb-1">
              <span className="text-xs font-medium text-slate-500">Shannon Entropy</span>
              <span className={`text-xs font-mono font-bold ${entropyPct > 50 ? 'text-amber-600' : 'text-emerald-600'}`}>
                {entropyPct}%
              </span>
            </div>
            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${entropyPct}%` }}
                transition={{ duration: 0.8 }}
                className={`h-full rounded-full ${entropyPct > 50 ? 'bg-amber-400' : 'bg-emerald-400'}`}
              />
            </div>
          </div>
        </div>

        {/* TTA Agreement */}
        {tta_agreement !== undefined && (
          <div className="flex items-center justify-between pt-2 border-t border-slate-50">
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-violet-500" />
              <span className="text-xs font-medium text-slate-500">TTA View Agreement</span>
            </div>
            <span className={`text-xs font-mono font-bold ${ttaAgreementPct < 70 ? 'text-amber-600' : 'text-emerald-600'}`}>
              {ttaAgreementPct}%
            </span>
          </div>
        )}

        {/* Localization stats */}
        {localization && (
          <div className="flex items-center justify-between pt-2 border-t border-slate-50">
            <div className="flex items-center gap-2">
              <Crosshair className="w-4 h-4 text-yellow-500" />
              <span className="text-xs font-medium text-slate-500">Tumor Region Area</span>
            </div>
            <span className="text-xs font-mono font-bold text-yellow-600">
              {localization.area_pct}%
            </span>
          </div>
        )}
      </div>

      {/* Class Probability Bars */}
      {all_class_probs && Object.keys(all_class_probs).length > 1 && (
        <div className="card space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 className="w-4 h-4 text-neural-600" />
            <h3 className="font-semibold text-slate-700 text-sm">Class Probability Distribution</h3>
          </div>
          {Object.entries(all_class_probs)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, prob]) => (
              <ClassProbBar key={cls} label={cls} prob={prob} maxProb={maxProb} />
            ))}
        </div>
      )}

      {/* Risk Assessment */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-600">Risk Assessment</span>
        <RiskBadge level={risk_level} />
      </div>

      {/* Heatmap / Explainability */}
      {heatmap_image && (
        <div className="card space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-neural-600" />
              <h3 className="font-semibold text-slate-700 text-sm">Explainability Maps</h3>
            </div>
            {/* Mode selector */}
            <div className="flex items-center gap-1 bg-slate-100 rounded-xl p-1">
              {[
                { id: 'gradcam', icon: Zap, label: 'Grad-CAM++' },
                { id: 'scorecam', icon: Activity, label: 'Score-CAM' },
                ...(comparison_strip ? [{ id: 'comparison', icon: GitCompare, label: 'Compare' }] : []),
              ].map(({ id, icon: Icon, label }) => (
                <button
                  key={id}
                  onClick={() => setHeatmapMode(id)}
                  className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    heatmapMode === id
                      ? 'bg-white text-neural-700 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  <Icon className="w-3 h-3" />
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Intensity slider (not for comparison mode) */}
          {heatmapMode !== 'comparison' && (
            <div className="flex items-center gap-3">
              <label className="text-[10px] uppercase font-bold text-slate-400 shrink-0">Intensity</label>
              <input
                type="range" min="0" max="1" step="0.01"
                value={opacity} onChange={e => setOpacity(parseFloat(e.target.value))}
                className="flex-1 accent-neural-500"
              />
            </div>
          )}

          {/* Bbox toggle — only when localization available and not comparison mode */}
          {localization && heatmapMode !== 'comparison' && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowBbox(v => !v)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-all ${
                  showBbox ? 'bg-yellow-50 border-yellow-300 text-yellow-700' : 'border-slate-200 text-slate-400'
                }`}
              >
                <Crosshair className="w-3 h-3" />
                Bounding Box
              </button>
              <span className="text-xs text-slate-400">Area: {localization.area_pct}% of scan</span>
            </div>
          )}

          <AnimatePresence mode="wait">
            <motion.div
              key={heatmapMode}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className={`relative rounded-xl overflow-hidden bg-slate-100 border border-slate-100 ${
                heatmapMode === 'comparison' ? 'aspect-[3/1]' : 'aspect-square'
              }`}
            >
              <img
                src={currentHeatmap}
                alt={heatmapMode === 'comparison' ? 'Original | Grad-CAM++ | Score-CAM' : `${heatmapMode} heatmap`}
                style={heatmapMode !== 'comparison' ? {
                  filter: `saturate(${opacity * 2}) brightness(${0.8 + opacity * 0.2})`
                } : {}}
                className="w-full h-full object-cover transition-all duration-300"
              />
              {showBbox && localization && heatmapMode !== 'comparison' && (
                <BboxCanvas localization={localization} />
              )}
            </motion.div>
          </AnimatePresence>

          <p className="text-xs text-slate-400">
            {heatmapMode === 'gradcam' && 'Grad-CAM++: gradient-weighted activation map highlighting decision regions.'}
            {heatmapMode === 'scorecam' && 'Score-CAM: perturbation-based, gradient-free — more faithful localisation.'}
            {heatmapMode === 'comparison' && 'Left: Original MRI · Centre: Grad-CAM++ · Right: Score-CAM comparison.'}
          </p>
        </div>
      )}

      {/* Clinical note */}
      <div className="card border-l-4 border-l-neural-400">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="w-4 h-4 text-neural-600" />
          <h3 className="font-semibold text-slate-700 text-sm">Clinical Note</h3>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-line">{clinical_note}</p>
      </div>

      {/* Recommendation */}
      <div className="card border-l-4 border-l-emerald-400">
        <div className="flex items-center gap-2 mb-2">
          <FileText className="w-4 h-4 text-emerald-600" />
          <h3 className="font-semibold text-slate-700 text-sm">Clinical Recommendations</h3>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-line">{recommendation}</p>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2 p-3 bg-slate-50 rounded-xl">
        <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-400 leading-relaxed">
          This AI system (v2: EfficientNet-B4 + TTA + MC Dropout + Score-CAM) is intended as a
          decision-support tool only. All results must be reviewed and confirmed by a qualified
          medical professional before any clinical action is taken.
        </p>
      </div>
    </div>
  )
}
