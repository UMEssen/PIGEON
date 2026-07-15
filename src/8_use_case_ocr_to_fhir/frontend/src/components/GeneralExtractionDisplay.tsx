import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { FileText, Activity, Pill, Stethoscope, Dna, AlertCircle } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';

interface GeneralExtractionDisplayProps {
  data: any;
}

const GeneralExtractionDisplay: React.FC<GeneralExtractionDisplayProps> = ({ data }) => {
  const { t } = useLanguage();
  if (!data) return null;

  const {
    document_type,
    patient_info,
    date,
    summary,
    diagnoses,
    medications,
    procedures,
    molecular_genetics,
    key_findings
  } = data;

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="text-2xl flex items-center gap-2">
                <FileText className="h-6 w-6 text-blue-500" />
                {document_type || t('medicalDocument')}
              </CardTitle>
              <p className="text-muted-foreground mt-1">
                {t('date')}: {date || t('na')}
              </p>
            </div>
            {patient_info && (
              <div className="text-right">
                <p className="font-semibold">{patient_info.name || t('unknownPatient')}</p>
                <p className="text-sm text-muted-foreground">{patient_info.dob || t('dobNa')}</p>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="bg-muted/30 p-4 rounded-md">
            <h4 className="font-semibold mb-2 flex items-center gap-2">
              <Activity className="h-4 w-4" /> {t('summary')}
            </h4>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {summary || t('noSummary')}
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Diagnoses */}
        {diagnoses && diagnoses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-red-500" /> {t('diagnoses')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {diagnoses.map((diagnosis: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-red-400 shrink-0" />
                    {diagnosis}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {/* Medications */}
        {medications && medications.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Pill className="h-5 w-5 text-green-500" /> {t('medications')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {medications.map((med: string, idx: number) => (
                  <Badge key={idx} variant="secondary" className="text-sm py-1">
                    {med}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Procedures & Key Findings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {procedures && procedures.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Stethoscope className="h-5 w-5 text-purple-500" /> {t('procedures')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {procedures.map((proc: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-purple-400 shrink-0" />
                    {proc}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {key_findings && key_findings.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="h-5 w-5 text-orange-500" /> {t('keyFindings')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {key_findings.map((finding: string, idx: number) => (
                  <li key={idx} className="flex items-start gap-2 text-sm">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-orange-400 shrink-0" />
                    {finding}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Molecular Genetics */}
      {molecular_genetics && (molecular_genetics.mutations?.length > 0 || molecular_genetics.details) && (
        <Card className="border-l-4 border-l-indigo-500">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Dna className="h-5 w-5 text-indigo-500" /> {t('molecularGenetics')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {molecular_genetics.mutations && molecular_genetics.mutations.length > 0 && (
                <div>
                  <h5 className="text-sm font-semibold mb-2">{t('detectedMutations')}</h5>
                  <div className="flex flex-wrap gap-2">
                    {molecular_genetics.mutations.map((mutation: string, idx: number) => (
                      <Badge key={idx} className="bg-indigo-100 text-indigo-800 hover:bg-indigo-200 border-indigo-200">
                        {mutation}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              {molecular_genetics.details && (
                <div>
                  <h5 className="text-sm font-semibold mb-1">{t('details')}</h5>
                  <p className="text-sm text-muted-foreground">{molecular_genetics.details}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default GeneralExtractionDisplay;
