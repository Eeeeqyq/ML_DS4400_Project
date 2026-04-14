import { SchoolInput } from "./predict";

export interface Preset {
  name: string;
  emoji: string;
  description: string;
  input: SchoolInput;
}

export const PRESETS: Preset[] = [
  {
    name: "State University",
    emoji: "🏫",
    description: "Large public school, moderate tuition, broad programs",
    input: {
      UGDS_log: Math.log1p(12000),
      ADM_RATE: 0.75,
      ADM_RATE_missing: 0,
      TUITIONFEE_OUT: 22000,
      NPT4: 11000,
      PCTPELL: 0.35,
      PCTFLOAN: 0.45,
      UGDS_WHITE: 0.55,
      UGDS_BLACK: 0.12,
      UGDS_HISP: 0.15,
      UGDS_ASIAN: 0.06,
      PCIP_reported: 1,
      high_earning_share: 0.30,
      CONTROL: "1",
      PREDDEG: "3",
      HIGHDEG: "4",
      LOCALE: "13.0",
      REGION: "5",
    },
  },
  {
    name: "Elite Private",
    emoji: "🎓",
    description: "Selective private nonprofit, high tuition, strong high-earning programs",
    input: {
      UGDS_log: Math.log1p(6000),
      ADM_RATE: 0.20,
      ADM_RATE_missing: 0,
      TUITIONFEE_OUT: 58000,
      NPT4: 22000,
      PCTPELL: 0.18,
      PCTFLOAN: 0.30,
      UGDS_WHITE: 0.40,
      UGDS_BLACK: 0.08,
      UGDS_HISP: 0.12,
      UGDS_ASIAN: 0.18,
      PCIP_reported: 1,
      high_earning_share: 0.45,
      CONTROL: "2",
      PREDDEG: "3",
      HIGHDEG: "4",
      LOCALE: "11.0",
      REGION: "1",
    },
  },
  {
    name: "For-Profit Trade School",
    emoji: "🏢",
    description: "Certificate-focused, open admission, vocational programs",
    input: {
      UGDS_log: Math.log1p(300),
      ADM_RATE: null,
      ADM_RATE_missing: 1,
      TUITIONFEE_OUT: 16000,
      NPT4: 18000,
      PCTPELL: 0.72,
      PCTFLOAN: 0.70,
      UGDS_WHITE: 0.30,
      UGDS_BLACK: 0.25,
      UGDS_HISP: 0.30,
      UGDS_ASIAN: 0.02,
      PCIP_reported: 0,
      high_earning_share: 0,
      CONTROL: "3",
      PREDDEG: "1",
      HIGHDEG: "1",
      LOCALE: "21.0",
      REGION: "5",
    },
  },
];

export const FIELD_LABELS: Record<string, string> = {
  UGDS_log: "Enrollment (students)",
  ADM_RATE: "Admission Rate",
  TUITIONFEE_OUT: "Tuition ($)",
  NPT4: "Net Price ($)",
  PCTPELL: "% on Pell Grants",
  PCTFLOAN: "% with Federal Loans",
  high_earning_share: "High-Earning Score",
  UGDS_WHITE: "% White",
  UGDS_BLACK: "% Black",
  UGDS_HISP: "% Hispanic",
  UGDS_ASIAN: "% Asian",
};

export const CONTROL_OPTIONS = [
  { value: "1", label: "Public" },
  { value: "2", label: "Private Nonprofit" },
  { value: "3", label: "For-Profit" },
];

export const PREDDEG_OPTIONS = [
  { value: "1", label: "Certificate" },
  { value: "2", label: "Associate's" },
  { value: "3", label: "Bachelor's" },
  { value: "4", label: "Graduate" },
];
