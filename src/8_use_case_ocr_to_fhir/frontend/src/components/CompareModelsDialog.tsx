import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Loader2 } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface CompareModelsDialogProps {
  open: boolean;
  onClose: () => void;
  medicalText: string;
}

const CompareModelsDialog: React.FC<CompareModelsDialogProps> = ({ open, onClose, medicalText }) => {
  const { t } = useLanguage();
  const [gemmaResult, setGemmaResult] = useState<any>(null);
  const [qwenResult, setQwenResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    setLoading(true);
    setError(null);
    setGemmaResult(null);
    setQwenResult(null);

    try {
      const { api } = await import('../services/api');
      const result = await api.compareExtract(medicalText);
      setGemmaResult(result.gemma_result);
      setQwenResult(result.qwen_result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const renderResult = (result: any, title: string, colorClass: string) => {
    if (!result) return null;

    const isJson = result && typeof result === 'object' && !result.error && !result.raw_content;
    
    return (
      <div className="flex-1 flex flex-col gap-2">
        <div className="flex items-center gap-2 mb-1">
          <h3 className={`text-lg font-semibold ${colorClass}`}>
            {title}
          </h3>
          <Badge variant={isJson ? "success" : "destructive"}>
            {isJson ? t('validJson') : t('plainText')}
          </Badge>
        </div>
        
        <div className="flex-1 p-3 bg-muted/50 border rounded-md overflow-auto max-h-[500px]">
          <pre className="m-0 whitespace-pre-wrap break-words text-xs font-mono">
            {isJson ? JSON.stringify(result, null, 2) : (result.raw_content || JSON.stringify(result, null, 2))}
          </pre>
        </div>
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={(val) => !val && onClose()}>
      <DialogContent className="max-w-[90vw] h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{t('compareModelsTitle')}</DialogTitle>
        </DialogHeader>
        
        <div className="flex justify-center my-4">
          <Button 
            onClick={handleCompare} 
            disabled={loading}
            className="bg-violet-600 hover:bg-violet-700"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> {t('runningComparison')}
              </>
            ) : (
              t('runComparison')
            )}
          </Button>
        </div>

        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-600 text-sm">
            {error}
          </div>
        )}

        <div className="flex-1 overflow-hidden flex gap-4">
          {loading ? (
            <div className="w-full h-full flex items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (gemmaResult || qwenResult) ? (
            <>
              {renderResult(gemmaResult, "Gemma 2 9B", "text-blue-600")}
              <div className="w-px bg-border" />
              {renderResult(qwenResult, "Qwen 2.5 14B", "text-violet-600")}
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-muted-foreground">
              Click "Run Comparison" to start
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CompareModelsDialog;
