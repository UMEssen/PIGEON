import React, { useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../services/api';
import BatchClassification from './BatchClassification';
import CompareModelsDialog from './CompareModelsDialog';
import GeneralExtractionDisplay from './GeneralExtractionDisplay';
import PipelineVisualizer from './PipelineVisualizer';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { cn } from '../lib/utils';
import { ChevronDown, ChevronUp, FileText, Upload, Activity, Sun, Moon, Languages } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';

type ShowcaseEvent = {
  event: string;
  [key: string]: any;
};

const ShowcasePage: React.FC = () => {
  const { theme, setTheme } = useTheme();
  const { language, setLanguage, t } = useLanguage();

  const [viewMode, setViewMode] = useState<'single' | 'batch'>('single');
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [fileText, setFileText] = useState<string>('');
  
  // Pigeon Mode state
  const [pigeonResult, setPigeonResult] = useState<any>(null);
  const [pigeonError, setPigeonError] = useState<string | null>(null);
  const [pigeonEvents, setPigeonEvents] = useState<ShowcaseEvent[]>([]);
  const [running, setRunning] = useState(false);
  
  const [fhirResources, setFhirResources] = useState<any[] | null>(null);
  const [fhirStatistics, setFhirStatistics] = useState<Record<string, number> | null>(null);
  const [fhirLoading, setFhirLoading] = useState(false);
  const [expandedResource, setExpandedResource] = useState<string | null>(null);
  const [compareModelsOpen, setCompareModelsOpen] = useState(false);

  // Text input mode states
  const [inputMode, setInputMode] = useState<'file' | 'text'>('file');
  const [textInput, setTextInput] = useState<string>('');
  const [textInferenceResult, setTextInferenceResult] = useState<any>(null);
  const [textInferenceLoading, setTextInferenceLoading] = useState(false);
  const [textInferenceError, setTextInferenceError] = useState<string | null>(null);

  const codeBox = (content: any, maxHeight = 240) => (
    <div className="p-3 bg-muted/50 border rounded-md overflow-auto" style={{ maxHeight }}>
      <pre className="m-0 whitespace-pre-wrap break-words text-sm font-mono leading-relaxed">
        {typeof content === 'string' ? content : JSON.stringify(content, null, 2)}
      </pre>
    </div>
  );

  const pigeonActiveStep = useMemo(() => {
    let step = 0;
    for (const ev of pigeonEvents) {
      const t = ev.event;
      if (t === 'upload_received') step = Math.max(step, 0);
      else if (t === 'ocr_complete') step = Math.max(step, 1);
      else if (t === 'classification_complete') step = Math.max(step, 1); // Classification happens after OCR
      else if (t === 'pigeon_complete' || t === 'general_extraction_complete') step = Math.max(step, 2);
      else if (t === 'rag_complete') step = Math.max(step, 3);
      else if (t === 'complete') step = 4;
    }
    return step;
  }, [pigeonEvents]);

  const numPages = useMemo(() => {
    const ocrEvent = pigeonEvents.find(e => e.event === 'ocr_complete');
    return ocrEvent?.num_pages;
  }, [pigeonEvents]);

  const handleStart = async (forceVlm: boolean = false) => {
    if (!file && inputMode === 'file') return;
    if (!textInput.trim() && inputMode === 'text') return;
    
    if (inputMode === 'text') {
      // Text input mode - direct inference
      setTextInferenceResult(null);
      setTextInferenceError(null);
      setTextInferenceLoading(true);
      try {
        const result = await api.inferFromText(textInput);
        setTextInferenceResult(result);
      } catch (e: any) {
        setTextInferenceError(e.response?.data?.detail || e.message || String(e));
      } finally {
        setTextInferenceLoading(false);
      }
      return;
    }
    
    // Pigeon Mode - Simple extraction with streaming
    setPigeonResult(null);
    setPigeonError(null);
    // Immediately show upload state
    setPigeonEvents([{ event: 'upload_received', message: 'Upload started...' }]);
    setRunning(true);
    try {
      // We checked !file above, so file is safe here
      const result = await api.pigeonExtract(file!, forceVlm, (evt) => {
        setPigeonEvents((prev) => {
            // Avoid duplicate upload_received if backend sends it
            if (evt.event === 'upload_received' && prev.some(e => e.event === 'upload_received')) {
                return prev;
            }
            return [...prev, evt];
        });
      });
      console.log('Pigeon Mode Result:', result);
      setPigeonResult(result);
      if (result.error) {
        setPigeonError(result.error);
      }
    } catch (e) {
      setPigeonError(String(e));
    } finally {
      setRunning(false);
    }
  };

  useEffect(() => {
    if (!file) {
      setFileUrl(null);
      setFileText('');
      return;
    }
    const obj = URL.createObjectURL(file);
    setFileUrl(obj);
    const isText = file.type.startsWith('text') || /\.txt$/i.test(file.name);
    if (isText) {
      const reader = new FileReader();
      reader.onload = (e) => setFileText(String(e.target?.result || ''));
      reader.readAsText(file, 'utf-8');
    } else {
      setFileText('');
    }
    return () => { try { URL.revokeObjectURL(obj); } catch {} };
  }, [file]);

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <div className="flex gap-4">
          <Button 
            variant={viewMode === 'single' ? 'default' : 'outline'}
            onClick={() => setViewMode('single')}
          >
            {t('singleDocProcessing')}
          </Button>
          <Button 
            variant={viewMode === 'batch' ? 'default' : 'outline'}
            onClick={() => setViewMode('batch')}
          >
            {t('batchClassification')}
          </Button>
        </div>

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setLanguage(language === 'en' ? 'de' : 'en')}
            title={language === 'en' ? 'Switch to German' : 'Zu Englisch wechseln'}
          >
            <Languages className="h-[1.2rem] w-[1.2rem]" />
            <span className="sr-only">Toggle language</span>
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
          >
            <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>
      </div>

      {viewMode === 'batch' ? (
        <BatchClassification />
      ) : (
        <>
          <div className="grid grid-cols-[420px_1fr] gap-4 min-h-[calc(100vh-16px)]">
        <div className="pb-10 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>{t('title')}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Input Mode Toggle */}
              <div className="flex gap-2">
                <Button
                  variant={inputMode === 'file' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setInputMode('file')}
                >
                  <Upload className="mr-2 h-4 w-4" /> {t('fileUpload')}
                </Button>
                <Button
                  variant={inputMode === 'text' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setInputMode('text')}
                  className={inputMode === 'text' ? 'bg-emerald-500 hover:bg-emerald-600' : ''}
                >
                  <FileText className="mr-2 h-4 w-4" /> {t('textInput')}
                </Button>
              </div>

              {inputMode === 'file' ? (
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">{t('input')}</h3>
                  <Input type="file" accept=".pdf,.txt,application/pdf,text/plain" onChange={(e) => setFile(e.target.files?.[0] || null)} />
                </div>
              ) : (
                <div className="space-y-2">
                  <h3 className="text-sm font-medium">{t('medicalTextGerman')}</h3>
                  <Textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder={t('pastePlaceholder')}
                    className="min-h-[200px] font-mono text-sm"
                  />
                </div>
              )}
              
              <div className="flex gap-2 mt-4">
                <Button 
                  className="flex-1" 
                  onClick={() => handleStart(false)} 
                  disabled={(inputMode === 'file' && !file) || (inputMode === 'text' && !textInput.trim()) || running || textInferenceLoading}
                >
                  {(running || textInferenceLoading) ? t('processing') : (inputMode === 'text' ? t('extractFromText') : t('extractAuto'))}
                </Button>
                
                {inputMode === 'file' && (
                  <Button 
                    className="flex-1 bg-violet-600 hover:bg-violet-700" 
                    onClick={() => handleStart(true)} 
                    disabled={!file || running || textInferenceLoading}
                  >
                    {running ? t('processing') : t('extractVlm')}
                  </Button>
                )}
              </div>
              {(running || textInferenceLoading) && (<Progress value={30} className="mt-4" />)}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <h3 className="text-sm font-medium mb-2">
                {t('pipelineProgress')}
              </h3>
              
              <PipelineVisualizer 
                events={pigeonEvents} 
                activeStep={pigeonActiveStep} 
                documentType={pigeonResult?.document_type || pigeonEvents.find(e => e.event === 'classification_complete')?.document_type}
              />
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <h3 className="text-sm font-medium mb-2">{t('inputPreview')}</h3>
              {file ? (
                <div className="mt-2">
                  <p className="text-sm mb-2">
                    {file.name} — {(file.size / 1024).toFixed(1)} KB
                    {numPages && <span> — {numPages} {t('pages')}</span>}
                  </p>
                  {( (/\.pdf$/i.test(file.name) || (file.type || '').includes('pdf')) && fileUrl ) ? (
                    <div className="h-[420px] border rounded-md overflow-hidden">
                      <iframe src={`${fileUrl}#toolbar=0&navpanes=0`} title={t('inputPdfPreview')} className="w-full h-full border-none" />
                    </div>
                  ) : fileText ? (
                    <>{codeBox(fileText, 360)}</>
                  ) : (
                    <p className="text-sm text-muted-foreground">{t('previewUnavailable')}</p>
                  )}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">{t('noFileSelected')}</p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          {/* Welcome / Explanation Card */}
          {!pigeonResult && !textInferenceResult && (
            <Card className="bg-gradient-to-br from-white to-slate-50 dark:from-slate-950 dark:to-slate-900 border-slate-200 dark:border-slate-800">
              <CardHeader>
                <CardTitle className="text-2xl text-primary">{t('welcomeTitle')}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-muted-foreground">
                <p>
                  {t('welcomeDesc')}
                </p>
                
                <div className="mt-6 space-y-6">
                  <div className="border-l-4 border-primary pl-4">
                    <h3 className="font-semibold text-foreground text-lg mb-2">{t('howItWorks')}</h3>
                    <p className="text-sm leading-relaxed">
                      {t('howItWorksDesc')}
                    </p>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <div className="bg-white dark:bg-slate-900 p-4 rounded-lg border shadow-sm">
                      <div className="flex items-center gap-2 mb-3">
                        <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">{t('caseA')}</Badge>
                        <h4 className="font-semibold text-foreground">{t('caseATitle')}</h4>
                      </div>
                      <p className="text-sm">
                        {t('caseADesc')}
                      </p>
                    </div>

                    <div className="bg-white dark:bg-slate-900 p-4 rounded-lg border shadow-sm">
                      <div className="flex items-center gap-2 mb-3">
                        <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">{t('caseB')}</Badge>
                        <h4 className="font-semibold text-foreground">{t('caseBTitle')}</h4>
                      </div>
                      <p className="text-sm">
                        {t('caseBDesc')}
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Text Inference Mode Results */}
          {inputMode === 'text' && textInferenceResult && (
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 mb-4">
                  <h2 className="text-lg font-semibold text-emerald-500">
                    {t('pigeonTextInferenceResults')}
                  </h2>
                  <Badge className="bg-emerald-500 hover:bg-emerald-600">
                    {textInferenceResult.model || "Pigeon"}
                  </Badge>
                </div>
                
                {textInferenceError && (
                  <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
                    <p className="text-sm text-red-600">{textInferenceError}</p>
                  </div>
                )}
                
                <div className="mb-6">
                  <h3 className="text-sm font-medium mb-2">{t('inputMedicalText')}</h3>
                  <div className="p-3 bg-muted/50 rounded-md max-h-[300px] overflow-auto">
                    <p className="whitespace-pre-wrap font-mono text-xs">
                      {textInferenceResult.medical_text}
                    </p>
                  </div>
                </div>
                
                <h3 className="text-sm font-medium mb-2 mt-6">
                  {t('extractedMedicalInfo')}
                </h3>
                {codeBox(textInferenceResult.extraction_result, 600)}
                
                {/* FHIR Conversion */}
                {textInferenceResult.extraction_result && !fhirResources && (
                  <div className="mt-6 text-center">
                    <Button
                      variant="secondary"
                      disabled={fhirLoading}
                      onClick={async () => {
                        setFhirLoading(true);
                        try {
                          const result = await api.convertToFHIR(textInferenceResult.extraction_result);
                          setFhirResources(result.resources || []);
                          setFhirStatistics(result.statistics || {});
                        } catch (e) {
                          console.error('FHIR conversion error:', e);
                        } finally {
                          setFhirLoading(false);
                        }
                      }}
                    >
                      {fhirLoading ? t('converting') : t('convertToFHIR')}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          
          {/* Pigeon Mode Results */}
          {inputMode === 'file' && pigeonResult && (
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 mb-4">
                  <h2 className="text-lg font-semibold text-emerald-500">
                    {t('medicalExtractionResults')}
                  </h2>
                  {pigeonResult.has_text_layer !== undefined && (
                    <Badge variant={pigeonResult.has_text_layer ? "default" : "secondary"}>
                      {pigeonResult.has_text_layer ? t('nativeText') : t('vlmOcr')}
                    </Badge>
                  )}
                  {pigeonResult.used_vlm_ocr && (
                    <Badge className="bg-violet-500 hover:bg-violet-600">
                      Qwen3-VL-30B
                    </Badge>
                  )}
                </div>
                
                {pigeonError && (
                  <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
                    <p className="text-sm text-red-600">{pigeonError}</p>
                  </div>
                )}
                
                {pigeonResult.medical_text && (
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-medium">{t('extractedMedicalText')}</h3>
                      {pigeonResult.document_type && (
                        <div className="flex flex-col items-end">
                          <Badge variant="outline" className="text-xs mb-1">
                            {t('type')}: {pigeonResult.document_type}
                          </Badge>
                          {pigeonResult.extraction_result?.explanation && (
                            <span className="text-[10px] text-muted-foreground max-w-[200px] text-right">
                              {pigeonResult.extraction_result.explanation}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="outline" className="w-full">
                          <FileText className="mr-2 h-4 w-4" />
                          {t('showExtractedText')}
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col">
                        <DialogHeader>
                          <DialogTitle>{t('extractedMedicalText')}</DialogTitle>
                        </DialogHeader>
                        <div className="flex-1 overflow-auto p-4 bg-muted/50 rounded-md mt-2 border">
                          <div className="prose prose-sm dark:prose-invert max-w-none">
                            <ReactMarkdown>{pigeonResult.medical_text}</ReactMarkdown>
                          </div>
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                )}
                
                {pigeonResult.document_type === 'Arztbrief' ? (
                  <>
                    <h3 className="text-sm font-medium mb-2 mt-6">
                      {t('structuredMedicalInfo')}
                    </h3>
                    {codeBox(pigeonResult.extraction_result, 600)}
                  </>
                ) : (
                  <div className="mt-6">
                    <GeneralExtractionDisplay data={pigeonResult.extraction_result} />
                  </div>
                )}
                
                {/* FHIR Conversion and Model Comparison Section */}
                {pigeonResult.extraction_result && !fhirResources && pigeonResult.document_type === 'Arztbrief' && (
                  <div className="mt-6 text-center flex gap-4 justify-center">
                    <Button 
                      className="bg-emerald-600 hover:bg-emerald-700"
                      onClick={async () => {
                        setFhirLoading(true);
                        try {
                          const result = await api.convertToFHIR(pigeonResult.extraction_result);
                          setFhirResources(result.resources);
                          setFhirStatistics(result.statistics);
                        } catch (e) {
                          console.error('FHIR conversion error:', e);
                          setPigeonError('FHIR conversion failed: ' + String(e));
                        } finally {
                          setFhirLoading(false);
                        }
                      }}
                      disabled={fhirLoading}
                    >
                      {fhirLoading ? t('converting') : t('convertToFHIRR5')}
                    </Button>
                    
                    <Button 
                      className="bg-violet-500 hover:bg-violet-600"
                      onClick={() => setCompareModelsOpen(true)}
                    >
                      {t('compareModels')}
                    </Button>
                  </div>
                )}
                
                {/* FHIR Resources Display */}
                {fhirResources && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold text-emerald-600 mb-4">
                      {t('fhirR5Resources')}
                    </h3>
                    
                    {/* Statistics */}
                    {fhirStatistics && (
                      <div className="mb-4 p-3 bg-emerald-50 border border-emerald-100 rounded-md">
                        <h4 className="text-sm font-medium mb-2">{t('generatedResources')}</h4>
                        <div className="flex gap-2 flex-wrap">
                          {Object.entries(fhirStatistics).map(([type, count]) => (
                            <Badge key={type} variant="outline" className="border-emerald-200 text-emerald-700 bg-emerald-50">
                              {type}: {count}
                            </Badge>
                          ))}
                          <Badge className="bg-emerald-600 hover:bg-emerald-700">
                            {t('total')}: {fhirResources.length}
                          </Badge>
                        </div>
                      </div>
                    )}
                    
                    {/* Resource List */}
                    <div className="flex flex-col gap-2">
                      {fhirResources.map((resource, idx) => (
                        <Card key={resource.id || idx}>
                          <CardContent className="p-4">
                            <div 
                              className="flex justify-between items-center cursor-pointer"
                              onClick={() => setExpandedResource(expandedResource === resource.id ? null : resource.id)}
                            >
                              <div>
                                <h4 className="text-sm font-medium">
                                  {resource.resourceType}
                                </h4>
                                <p className="text-xs text-muted-foreground">
                                  ID: {resource.id}
                                </p>
                              </div>
                              <Button variant="ghost" size="sm">
                                {expandedResource === resource.id ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                                <span className="ml-1">{expandedResource === resource.id ? t('hideJson') : t('showJson')}</span>
                              </Button>
                            </div>
                            
                            {expandedResource === resource.id && (
                              <div className="mt-4">
                                {codeBox(resource, 400)}
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
          </div>
        </div>

        {/* Compare Models Dialog */}
        {pigeonResult?.medical_text && (
          <CompareModelsDialog
            open={compareModelsOpen}
            onClose={() => setCompareModelsOpen(false)}
            medicalText={pigeonResult.medical_text}
          />
        )}
        </>
      )}
    </div>
  );
};

export default ShowcasePage;
