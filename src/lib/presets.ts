import { TransactionType } from "@/types/transaction";

export interface TransactionPreset {
  id: string;
  name: string;
  description: string;
  expectedOutcome: "APPROVE" | "STEP_UP" | "BLOCK";
  type: TransactionType;
  amount: number;
  originAccountIndex: number; // Index in DEFAULT_ORIGIN_ACCOUNTS
  customNameDest?: string;
}

export const TRANSACTION_PRESETS: TransactionPreset[] = [
  {
    id: "normal_payment",
    name: "Normal PAYMENT (low risk)",
    description: "Small payment to merchant - should be approved",
    expectedOutcome: "APPROVE",
    type: "PAYMENT",
    amount: 5000,
    originAccountIndex: 1, // Account with 150000 balance
  },
  {
    id: "large_transfer",
    name: "Large TRANSFER draining balance (step-up)",
    description: "Transfer that drains most of the balance - requires verification",
    expectedOutcome: "STEP_UP",
    type: "TRANSFER",
    amount: 140000,
    originAccountIndex: 1, // Account with 150000 balance
  },
  {
    id: "suspicious_cashout",
    name: "CASH_OUT near empty balance (block)",
    description: "Cash out from near-zero balance - should be blocked",
    expectedOutcome: "BLOCK",
    type: "CASH_OUT",
    amount: 200000,
    originAccountIndex: 3, // Account with 25000 balance
  },
  {
    id: "large_cashin",
    name: "CASH_IN unusually large (step-up)",
    description: "Large cash deposit - may require verification",
    expectedOutcome: "STEP_UP",
    type: "CASH_IN",
    amount: 180000,
    originAccountIndex: 4, // Account with 0 balance
  },
  {
    id: "small_debit",
    name: "Small DEBIT fee (low risk)",
    description: "Bank fee deduction - should be approved",
    expectedOutcome: "APPROVE",
    type: "DEBIT",
    amount: 500,
    originAccountIndex: 0, // Account with 50000 balance
  },
  {
    id: "known_fraud_pattern",
    name: "Known Fraud Pattern (high risk)",
    description: "Suspicious 3 AM transfer with balance drain - should be blocked",
    expectedOutcome: "BLOCK",
    type: "TRANSFER",
    amount: 99000,
    originAccountIndex: 0, // Account with 50000 balance (will trigger multiple risk factors)
    customNameDest: "C9999999999", // New suspicious destination
  },
];
