import { useState, useEffect, useMemo, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { PresetButtons } from "./PresetButtons";
import { BalanceDisplay } from "./BalanceDisplay";
import { TimeStepBadge } from "@/components/ui/TimeStepBadge";
import { TransactionPreset } from "@/lib/presets";
import { EVENT_TYPE_LABELS, formatCurrency } from "@/lib/eventTypes";
import { 
  TransactionType, 
  Transaction, 
  TRANSACTION_TYPES, 
  DEFAULT_ORIGIN_ACCOUNTS 
} from "@/types/transaction";
import {
  getLastStep,
  setLastStep,
  getOriginAccounts,
  getDestinationBalances,
  updateDestinationBalance,
  updateOriginAccount,
  saveTransaction,
  getAdminSettings,
  setPendingTransaction,
} from "@/lib/storage";
import { computeIsFlaggedFraud, scoreTransaction } from "@/lib/scoring";
import { RotateCcw, Send, ChevronDown, ChevronUp, DollarSign } from "lucide-react";

interface FormErrors {
  [key: string]: string;
}

export function TransactionForm() {
  const navigate = useNavigate();
  const formRef = useRef<HTMLFormElement>(null);
  const errorSummaryRef = useRef<HTMLDivElement>(null);

  // Form state
  const [step, setStep] = useState(getLastStep() + 1);
  const [type, setType] = useState<TransactionType | "">("");
  const [nameOrig, setNameOrig] = useState("");
  const [amount, setAmount] = useState("");
  const [nameDest, setNameDest] = useState("");
  const [advancedMode, setAdvancedMode] = useState(false);
  const [allowNegativeBalance, setAllowNegativeBalance] = useState(false);
  const [manualOldBalanceDest, setManualOldBalanceDest] = useState("");
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const originAccounts = useMemo(() => getOriginAccounts(), []);
  const destBalances = useMemo(() => getDestinationBalances(), []);
  const adminSettings = useMemo(() => getAdminSettings(), []);

  // Get selected origin account
  const selectedOrigin = originAccounts.find(a => a.id === nameOrig);
  const oldbalanceOrg = selectedOrigin?.balance ?? 0;

  // Compute destination name based on type
  const computedNameDest = useMemo(() => {
    if (advancedMode && nameDest) return nameDest;
    if (!type) return "";
    
    switch (type) {
      case "CASH OUT":
      case "CASH IN":
        return "CASH AGENT";
      case "DEBIT":
        return "BANK FEE ACCOUNT";
      case "PAYMENT":
        return `M${Math.random().toString().slice(2, 12)}`;
      case "TRANSFER":
        return `C${Math.random().toString().slice(2, 12)}`;
      default:
        return "";
    }
  }, [type, advancedMode, nameDest]);

  // Compute balances
  const amountNum = parseFloat(amount) || 0;
  
  const oldbalanceDest = advancedMode && manualOldBalanceDest !== ""
    ? parseFloat(manualOldBalanceDest) || 0
    : destBalances[computedNameDest] ?? 0;

  const newbalanceOrig = useMemo(() => {
    if (type === "CASH IN") {
      return oldbalanceOrg + amountNum;
    }
    return Math.max(oldbalanceOrg - amountNum, 0);
  }, [type, oldbalanceOrg, amountNum]);

  const newbalanceDest = useMemo(() => {
    // For simplicity, CASH OUT doesn't reduce agent balance
    if (type === "CASH OUT") {
      return oldbalanceDest;
    }
    return oldbalanceDest + amountNum;
  }, [type, oldbalanceDest, amountNum]);

  // Validation
  const validate = (): boolean => {
    const newErrors: FormErrors = {};

    if (step < 1) {
      newErrors.step = "Time step must be at least 1";
    }
    if (!type) {
      newErrors.type = "Please select an event type";
    }
    if (!nameOrig) {
      newErrors.nameOrig = "Please select a sender account";
    }
    if (!amount || amountNum <= 0) {
      newErrors.amount = "Amount must be greater than $0";
    }
    if (
      !allowNegativeBalance && 
      adminSettings.blockInsufficientBalance &&
      type !== "CASH IN" && 
      amountNum > oldbalanceOrg
    ) {
      newErrors.amount = `Insufficient balance. Available: ${formatCurrency(oldbalanceOrg)}`;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Focus first error field
  useEffect(() => {
    if (Object.keys(errors).length > 0 && errorSummaryRef.current) {
      errorSummaryRef.current.focus();
    }
  }, [errors]);

  const handlePresetSelect = (preset: TransactionPreset) => {
    const account = DEFAULT_ORIGIN_ACCOUNTS[preset.originAccountIndex];
    setType(preset.type);
    setNameOrig(account.id);
    setAmount(preset.amount.toString());
    if (preset.customNameDest) {
      setAdvancedMode(true);
      setNameDest(preset.customNameDest);
    }
    setErrors({});
  };

  const handleReset = () => {
    setStep(getLastStep() + 1);
    setType("");
    setNameOrig("");
    setAmount("");
    setNameDest("");
    setAdvancedMode(false);
    setAllowNegativeBalance(false);
    setManualOldBalanceDest("");
    setErrors({});
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validate()) return;

    setIsSubmitting(true);

    try {
      const finalNameDest = computedNameDest;
      const isFlaggedFraud = computeIsFlaggedFraud(type as TransactionType, amountNum, adminSettings);

      const scoringResult = scoreTransaction(
        {
          type: type as TransactionType,
          amount: amountNum,
          oldbalanceOrg,
          newbalanceOrig: newbalanceOrig,
          oldbalanceDest,
          newbalanceDest,
          isFlaggedFraud,
        },
        adminSettings
      );

      const transaction: Transaction = {
        id: crypto.randomUUID(),
        step,
        type: type as TransactionType,
        amount: amountNum,
        nameOrig,
        oldbalanceOrg,
        newbalanceOrig,
        nameDest: finalNameDest,
        oldbalanceDest,
        newbalanceDest,
        isFraud: 0, // Default, can be labeled later
        isFlaggedFraud,
        riskScore: scoringResult.riskScore,
        decision: scoringResult.decision,
        reasons: scoringResult.reasons,
        createdAt: new Date().toISOString(),
      };

      // Save transaction
      saveTransaction(transaction);
      
      // Update balances
      updateOriginAccount(nameOrig, newbalanceOrig);
      updateDestinationBalance(finalNameDest, newbalanceDest);
      
      // Update last step
      setLastStep(step);

      // Store pending transaction for result page
      setPendingTransaction(transaction);

      // Navigate to result
      navigate("/result");
    } finally {
      setIsSubmitting(false);
    }
  };

  const isValid = type && nameOrig && amountNum > 0 && step >= 1;

  return (
    <form ref={formRef} onSubmit={handleSubmit} className="space-y-6" noValidate>
      {/* Error Summary */}
      {Object.keys(errors).length > 0 && (
        <div
          ref={errorSummaryRef}
          className="bg-danger-muted border border-danger/20 rounded-lg p-4"
          role="alert"
          aria-live="polite"
          tabIndex={-1}
        >
          <p className="font-medium text-danger mb-2">Please fix the following errors:</p>
          <ul className="list-disc list-inside text-sm text-danger space-y-1">
            {Object.entries(errors).map(([field, message]) => (
              <li key={field}>{message}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left Column - Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Time & Type */}
          <fieldset className="form-fieldset">
            <legend className="form-legend">Time & Event</legend>
            
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="step">Time Step (Hour)</Label>
                <Input
                  id="step"
                  type="number"
                  min={1}
                  value={step}
                  onChange={e => setStep(parseInt(e.target.value) || 1)}
                  aria-describedby={errors.step ? "step-error" : "step-hint"}
                  aria-invalid={!!errors.step}
                />
                {errors.step ? (
                  <p id="step-error" className="text-sm text-danger">{errors.step}</p>
                ) : (
                  <TimeStepBadge step={step} className="mt-1" />
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="type">Event Type</Label>
                <Select value={type} onValueChange={(v) => setType(v as TransactionType)}>
                  <SelectTrigger 
                    id="type"
                    aria-describedby={errors.type ? "type-error" : undefined}
                    aria-invalid={!!errors.type}
                  >
                    <SelectValue placeholder="Select event type..." />
                  </SelectTrigger>
                  <SelectContent>
                    {TRANSACTION_TYPES.map(t => (
                      <SelectItem key={t.value} value={t.value}>
                        <span className="font-medium">{EVENT_TYPE_LABELS[t.value]}</span>
                        <span className="text-muted-foreground ml-2 text-xs">({t.value})</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.type && (
                  <p id="type-error" className="text-sm text-danger">{errors.type}</p>
                )}
              </div>
            </div>
          </fieldset>

          {/* Parties */}
          <fieldset className="form-fieldset">
            <legend className="form-legend">Parties</legend>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="nameOrig">Sender Account</Label>
                <Select value={nameOrig} onValueChange={setNameOrig}>
                  <SelectTrigger 
                    id="nameOrig"
                    aria-describedby={errors.nameOrig ? "nameOrig-error" : undefined}
                    aria-invalid={!!errors.nameOrig}
                  >
                    <SelectValue placeholder="Select sender account..." />
                  </SelectTrigger>
                  <SelectContent>
                    {originAccounts.map(account => (
                      <SelectItem key={account.id} value={account.id}>
                        <span className="font-mono">{account.name}</span>
                        <span className="text-muted-foreground ml-2">
                          (Avail: {formatCurrency(account.balance)})
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.nameOrig && (
                  <p id="nameOrig-error" className="text-sm text-danger">{errors.nameOrig}</p>
                )}
                {selectedOrigin && (
                  <p className="text-sm text-muted-foreground">
                    Current balance: <span className="font-mono font-medium">{formatCurrency(oldbalanceOrg)}</span>
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="nameDest">Recipient / Merchant</Label>
                {advancedMode ? (
                  <Input
                    id="nameDest"
                    value={nameDest}
                    onChange={e => setNameDest(e.target.value)}
                    placeholder={computedNameDest || "Enter recipient..."}
                  />
                ) : (
                  <Input
                    id="nameDest"
                    value={computedNameDest}
                    readOnly
                    className="bg-muted font-mono"
                    aria-describedby="nameDest-hint"
                  />
                )}
                <p id="nameDest-hint" className="text-xs text-muted-foreground">
                  {advancedMode ? "Enter custom recipient" : "Auto-generated based on event type"}
                </p>
              </div>
            </div>
          </fieldset>

          {/* Amount */}
          <fieldset className="form-fieldset">
            <legend className="form-legend">Transaction Amount</legend>
            
            <div className="space-y-2">
              <Label htmlFor="amount">Amount</Label>
              <div className="relative">
                <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  id="amount"
                  type="number"
                  min={0}
                  step="0.01"
                  value={amount}
                  onChange={e => setAmount(e.target.value)}
                  placeholder="0.00"
                  className="pl-9"
                  aria-describedby={errors.amount ? "amount-error" : "amount-hint"}
                  aria-invalid={!!errors.amount}
                />
              </div>
              {errors.amount ? (
                <p id="amount-error" className="text-sm text-danger">{errors.amount}</p>
              ) : (
                <p id="amount-hint" className="text-xs text-muted-foreground">
                  Enter the transaction amount in USD (simulated)
                </p>
              )}
            </div>
          </fieldset>

          {/* Derived Balances */}
          {(nameOrig && amountNum > 0) && (
            <fieldset className="form-fieldset">
              <legend className="form-legend">Balance Changes</legend>
              <div className="space-y-2">
                <BalanceDisplay
                  label="Sender"
                  oldBalance={oldbalanceOrg}
                  newBalance={newbalanceOrig}
                  highlight={newbalanceOrig < oldbalanceOrg ? "decrease" : newbalanceOrig > oldbalanceOrg ? "increase" : "none"}
                />
                <BalanceDisplay
                  label="Recipient"
                  oldBalance={oldbalanceDest}
                  newBalance={newbalanceDest}
                  highlight={newbalanceDest > oldbalanceDest ? "increase" : "none"}
                />
              </div>
            </fieldset>
          )}

          {/* Advanced Mode */}
          <div className="section-card">
            <button
              type="button"
              className="flex items-center justify-between w-full text-left"
              onClick={() => setAdvancedMode(!advancedMode)}
              aria-expanded={advancedMode}
              aria-controls="advanced-options"
            >
              <span className="font-medium">Advanced Options</span>
              {advancedMode ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </button>

            {advancedMode && (
              <div id="advanced-options" className="mt-4 space-y-4 pt-4 border-t border-border">
                <div className="space-y-2">
                  <Label htmlFor="manualOldBalanceDest">Override Recipient Starting Balance</Label>
                  <div className="relative">
                    <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      id="manualOldBalanceDest"
                      type="number"
                      min={0}
                      value={manualOldBalanceDest}
                      onChange={e => setManualOldBalanceDest(e.target.value)}
                      placeholder={oldbalanceDest.toString()}
                      className="pl-9"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="allowNegative">Allow insufficient balance</Label>
                    <p className="text-xs text-muted-foreground">
                      Bypass balance check for testing edge cases
                    </p>
                  </div>
                  <Switch
                    id="allowNegative"
                    checked={allowNegativeBalance}
                    onCheckedChange={setAllowNegativeBalance}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Presets */}
        <div className="lg:col-span-1">
          <PresetButtons onSelect={handlePresetSelect} disabled={isSubmitting} />
        </div>
      </div>

      {/* Sticky Bottom Bar */}
      <div className="sticky-bottom-bar">
        <div className="container flex gap-3 justify-end">
          <Button
            type="button"
            variant="outline"
            onClick={handleReset}
            disabled={isSubmitting}
          >
            <RotateCcw className="h-4 w-4 mr-2" aria-hidden="true" />
            Reset
          </Button>
          <Button
            type="submit"
            disabled={!isValid || isSubmitting}
            className="min-w-[160px]"
          >
            {isSubmitting ? (
              "Processing..."
            ) : (
              <>
                <Send className="h-4 w-4 mr-2" aria-hidden="true" />
                Submit Transaction
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Spacer for sticky bar */}
      <div className="h-20" aria-hidden="true" />
    </form>
  );
}
