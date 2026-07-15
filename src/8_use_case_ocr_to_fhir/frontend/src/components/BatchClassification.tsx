import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Loader2, Play } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface ClassificationResult {
  document_type: string;
  confidence: number;
  reasoning: string;
}

interface FileResult {
  filename: string;
  runs: ClassificationResult[];
}

const BatchClassification: React.FC = () => {
  const { t } = useLanguage();
  const [files, setFiles] = useState<string[]>([]);
  const [results, setResults] = useState<Record<string, FileResult>>({});
  const [loading, setLoading] = useState(false);
  const [numRuns, setNumRuns] = useState(1);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const fileList = await api.getBatchFiles();
      setFiles(fileList);
      // Initialize results structure
      const initialResults: Record<string, FileResult> = {};
      fileList.forEach(f => {
        initialResults[f] = { filename: f, runs: [] };
      });
      setResults(initialResults);
    } catch (e) {
      console.error("Failed to load batch files", e);
    }
  };

  const runBatch = async () => {
    setLoading(true);
    const totalOps = files.length * numRuns;
    let completedOps = 0;
    setProgress({ current: 0, total: totalOps });

    // Reset results for new run
    const newResults = { ...results };
    files.forEach(f => {
      newResults[f].runs = [];
    });
    setResults(newResults);

    for (let run = 0; run < numRuns; run++) {
      for (const file of files) {
        try {
          const res = await api.runBatchClassification(file);
          setResults(prev => {
            const updated = { ...prev };
            updated[file].runs = [...updated[file].runs, res.classification];
            return updated;
          });
        } catch (e) {
          console.error(`Failed run ${run + 1} for ${file}`, e);
        }
        completedOps++;
        setProgress({ current: completedOps, total: totalOps });
      }
    }
    setLoading(false);
    setProgress(null);
  };

  return (
    <Card className="mt-8">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>{t('batchClassificationTest')}</span>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{t('runs')}</span>
              <Input 
                type="number" 
                min={1} 
                max={5} 
                value={numRuns} 
                onChange={(e) => setNumRuns(parseInt(e.target.value) || 1)}
                className="w-20 h-8"
              />
            </div>
            <Button onClick={runBatch} disabled={loading || files.length === 0}>
              {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              {t('runBatch')}
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {progress && (
          <div className="mb-4 text-sm text-muted-foreground">
            {t('processingProgress')} {progress.current} / {progress.total}
          </div>
        )}
        
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[300px]">{t('filename')}</TableHead>
                {Array.from({ length: numRuns }).map((_, i) => (
                  <TableHead key={i}>{t('run')} {i + 1}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {files.map((file) => (
                <TableRow key={file}>
                  <TableCell className="font-mono text-xs truncate max-w-[300px]" title={file}>
                    {file}
                  </TableCell>
                  {Array.from({ length: numRuns }).map((_, i) => {
                    const run = results[file]?.runs[i];
                    return (
                      <TableCell key={i}>
                        {run ? (
                          <div className="space-y-1">
                            <Badge variant={run.document_type === 'Arztbrief' ? 'default' : 'secondary'}>
                              {run.document_type}
                            </Badge>
                            <div className="text-xs text-muted-foreground">
                              {t('conf')} {(run.confidence * 100).toFixed(0)}%
                            </div>
                            <div className="text-[10px] text-muted-foreground leading-tight max-w-[200px]">
                              {run.reasoning}
                            </div>
                          </div>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
};

export default BatchClassification;
