export interface Patient {
  id?: string;
  name?: string;
  dob?: string;
  sex?: string;
}

export interface Document {
  path: string;
  key: string;
  category: string;
  date: string;
  type: 'pdf' | 'txt';
  status: ProcessingStatus;
  patient_number: string;
}

export enum ProcessingStatus {
  UNPROCESSED = 'unprocessed',
  QUEUED = 'queued',
  PARSING = 'parsing',
  EXTRACTING = 'extracting',
  SUMMARIZING = 'summarizing',
  DONE = 'done',
  ERROR = 'error',
}

export interface Finding {
  section: string;
  text: string;
  page: number;
}

export interface Measurement {
  name: string;
  value: string;
  unit: string;
  ref_range?: string;
  page: number;
}

export interface Code {
  system: string;
  code: string;
  page: number;
}

export interface Signature {
  signer: string;
  role: string;
  date?: string;
  page: number;
}

export interface ExtractionResult {
  document_type: string;
  document_date?: string;
  patient: Patient;
  provenance: {
    pages: Array<{ page: number; sections: string[] }>;
    figures: Array<{ page: number; caption: string; id: string }>;
  };
  findings: Finding[];
  measurements: Measurement[];
  codes: Code[];
  signatures: Signature[];
}

export interface ProcessingResult {
  document: Document;
  extraction?: ExtractionResult;
  summary?: string;
  error?: string;
  provenance_data?: any;
  parsed_html?: string;
  extracted_images?: string[];
  logs?: any[];
}

export interface WebSocketMessage {
  type: 'init' | 'document_complete' | 'complete' | 'error';
  document?: Document;
  summary?: string;
  extraction?: ExtractionResult;
  error?: string;
  provenance?: any;
  patient_number?: string;
  session_id?: string;
}
