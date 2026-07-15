import React from 'react';
import { cn } from '../lib/utils';
import { CheckCircle2, Circle, ArrowRight, GitBranch, FileText, BrainCircuit, Database, FileJson } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface PipelineVisualizerProps {
  events: any[];
  activeStep: number;
  documentType?: string;
}

const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({ events, activeStep, documentType }) => {
  const { t } = useLanguage();
  // Determine current state based on events
  const hasStarted = events.length > 0;
  const ocrComplete = events.some(e => e.event === 'ocr_complete');
  const classificationComplete = events.some(e => e.event === 'classification_complete');
  const isArztbrief = documentType === 'Arztbrief';
  const isOther = documentType && documentType !== 'Arztbrief';
  
  const pigeonComplete = events.some(e => e.event === 'pigeon_complete');
  const ragComplete = events.some(e => e.event === 'rag_complete');
  const generalComplete = events.some(e => e.event === 'general_extraction_complete');
  const isComplete = events.some(e => e.event === 'complete');

  const StepNode = ({ 
    active, 
    completed, 
    icon: Icon, 
    label, 
    sublabel 
  }: { 
    active: boolean; 
    completed: boolean; 
    icon: any; 
    label: string; 
    sublabel?: string 
  }) => (
    <div className={cn(
      "flex flex-col items-center z-10 transition-all duration-500",
      active ? "scale-110" : "opacity-70"
    )}>
      <div className={cn(
        "w-12 h-12 rounded-full flex items-center justify-center border-2 shadow-sm transition-colors duration-300",
        completed ? "bg-emerald-100 border-emerald-500 text-emerald-600" :
        active ? "bg-blue-100 border-blue-500 text-blue-600 animate-pulse" :
        "bg-slate-50 border-slate-200 text-slate-300"
      )}>
        {completed ? <CheckCircle2 className="w-6 h-6" /> : <Icon className="w-6 h-6" />}
      </div>
      <div className="mt-2 text-center">
        <p className={cn("text-xs font-semibold", active || completed ? "text-foreground" : "text-muted-foreground")}>{label}</p>
        {sublabel && <p className="text-[10px] text-muted-foreground">{sublabel}</p>}
      </div>
    </div>
  );

  const Connector = ({ active, vertical = false }: { active: boolean, vertical?: boolean }) => (
    <div className={cn(
      "transition-colors duration-500",
      vertical ? "w-0.5 h-8" : "h-0.5 flex-1 mx-2",
      active ? "bg-blue-400" : "bg-slate-200"
    )} />
  );

  return (
    <div className="py-6 px-2">
      {/* Main Trunk */}
      <div className="flex items-center justify-center mb-8">
        <StepNode 
          active={hasStarted && !ocrComplete} 
          completed={ocrComplete} 
          icon={FileText} 
          label={t('upload')} 
        />
        <Connector active={ocrComplete} />
        <StepNode 
          active={ocrComplete && !classificationComplete} 
          completed={classificationComplete} 
          icon={BrainCircuit} 
          label={t('ocrAnalysis')} 
        />
        <Connector active={classificationComplete} />
        <StepNode 
          active={classificationComplete && !isArztbrief && !isOther} 
          completed={!!documentType} 
          icon={GitBranch} 
          label={t('classification')} 
          sublabel={documentType}
        />
      </div>

      {/* Branches */}
      <div className="grid grid-cols-2 gap-8 relative">
        {/* Connecting Lines from Classification */}
        {documentType && (
          <>
             <div className={cn(
               "absolute top-[-32px] left-1/2 w-[calc(25%+1rem)] h-8 border-t-2 border-l-2 rounded-tl-xl -translate-x-full",
               isArztbrief ? "border-blue-400" : "border-slate-200"
             )} />
             <div className={cn(
               "absolute top-[-32px] left-1/2 w-[calc(25%+1rem)] h-8 border-t-2 border-r-2 rounded-tr-xl",
               isOther ? "border-blue-400" : "border-slate-200"
             )} />
          </>
        )}

        {/* Left Branch: Pigeon */}
        <div className={cn("flex flex-col items-center space-y-4", !isArztbrief && "opacity-30 grayscale")}>
          <div className="bg-blue-50 px-3 py-1 rounded-full text-xs font-medium text-blue-700 mb-2">
            {t('arztbriefPipeline')}
          </div>
          <StepNode 
            active={isArztbrief && !pigeonComplete} 
            completed={pigeonComplete} 
            icon={Database} 
            label={t('pigeonExtraction')} 
          />
          <Connector active={pigeonComplete} vertical />
          <StepNode 
            active={isArztbrief && pigeonComplete && !ragComplete} 
            completed={ragComplete} 
            icon={CheckCircle2} 
            label={t('ragValidation')} 
          />
        </div>

        {/* Right Branch: General */}
        <div className={cn("flex flex-col items-center space-y-4", !isOther && "opacity-30 grayscale")}>
          <div className="bg-purple-50 px-3 py-1 rounded-full text-xs font-medium text-purple-700 mb-2">
            {t('generalPipeline')}
          </div>
          <StepNode 
            active={isOther && !generalComplete} 
            completed={generalComplete} 
            icon={BrainCircuit} 
            label={t('generalAgent')} 
          />
          <Connector active={generalComplete} vertical />
          <StepNode 
            active={isOther && generalComplete} 
            completed={generalComplete} 
            icon={FileJson} 
            label={t('summaryData')} 
          />
        </div>
      </div>
      
      {/* Final Status */}
      {isComplete && (
        <div className="mt-8 text-center animate-in fade-in slide-in-from-bottom-4">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-100 text-emerald-800 rounded-full font-medium text-sm">
            <CheckCircle2 className="w-4 h-4" />
            {t('processingComplete')}
          </div>
        </div>
      )}
    </div>
  );
};

export default PipelineVisualizer;
