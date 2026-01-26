import { useEffect, useState, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import { RiskScore } from "@/components/ui/RiskScore";
import { OTPChallenge } from "@/components/result/OTPChallenge";
import { TransactionDatasetView } from "@/components/result/TransactionDatasetView";
import { Transaction, Decision } from "@/types/transaction";
import { getPendingTransaction, clearPendingTransaction, updateTransaction } from "@/lib/storage";
import { PlayCircle, History, AlertCircle } from "lucide-react";

export default function Result() {
  const navigate = useNavigate();
  const [transaction, setTransaction] = useState<Transaction | null>(null);
  const [showOTP, setShowOTP] = useState(false);
  const [finalDecision, setFinalDecision] = useState<Decision | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    const pending = getPendingTransaction();
    if (!pending) {
      navigate("/simulate");
      return;
    }
    setTransaction(pending);
    setFinalDecision(pending.decision);
    
    if (pending.decision === "STEP_UP") {
      setShowOTP(true);
    }

    clearPendingTransaction();

    // Focus heading for accessibility
    setTimeout(() => {
      headingRef.current?.focus();
    }, 100);
  }, [navigate]);

  const handleOTPSuccess = () => {
    if (!transaction) return;
    const newDecision: Decision = "APPROVE_AFTER_STEPUP";
    setFinalDecision(newDecision);
    setShowOTP(false);
    updateTransaction(transaction.id, { decision: newDecision });
  };

  const handleOTPFail = () => {
    if (!transaction) return;
    const newDecision: Decision = "BLOCK_STEPUP_FAILED";
    setFinalDecision(newDecision);
    setShowOTP(false);
    updateTransaction(transaction.id, { decision: newDecision });
  };

  if (!transaction) {
    return (
      <Layout>
        <div className="container py-12 text-center">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </Layout>
    );
  }

  const displayDecision = finalDecision || transaction.decision;

  const decisionColors = {
    APPROVE: "bg-success-muted border-success/20",
    STEP_UP: "bg-warning-muted border-warning/20",
    BLOCK: "bg-danger-muted border-danger/20",
    APPROVE_AFTER_STEPUP: "bg-success-muted border-success/20",
    BLOCK_STEPUP_FAILED: "bg-danger-muted border-danger/20",
  };

  return (
    <Layout>
      <div className="container py-6 sm:py-8">
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Decision Banner */}
          <div className={`rounded-lg border p-6 text-center ${decisionColors[displayDecision]}`}>
            <h1 
              ref={headingRef}
              tabIndex={-1}
              className="text-2xl sm:text-3xl font-bold mb-4 outline-none"
            >
              Transaction Result
            </h1>
            <DecisionBadge decision={displayDecision} size="lg" />
          </div>

          {/* OTP Challenge */}
          {showOTP && (
            <div className="section-card-elevated">
              <OTPChallenge onSuccess={handleOTPSuccess} onFail={handleOTPFail} />
            </div>
          )}

          {/* Risk Score */}
          {!showOTP && (
            <div className="section-card">
              <h2 className="font-semibold mb-4">Risk Analysis</h2>
              <RiskScore score={transaction.riskScore} size="lg" />
              
              <div className="mt-4 pt-4 border-t border-border">
                <h3 className="text-sm font-medium text-muted-foreground mb-2">Risk Factors</h3>
                <ul className="space-y-2">
                  {transaction.reasons.map((reason, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <AlertCircle className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" aria-hidden="true" />
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Transaction Summary */}
          {!showOTP && (
            <div className="section-card">
              <h2 className="font-semibold mb-4">Transaction Summary</h2>
              <dl className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <dt className="text-muted-foreground">Type</dt>
                  <dd className="font-medium">{transaction.type}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Amount</dt>
                  <dd className="font-mono font-medium">{transaction.amount.toLocaleString()}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Origin</dt>
                  <dd className="font-mono text-xs">{transaction.nameOrig}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Destination</dt>
                  <dd className="font-mono text-xs">{transaction.nameDest}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Flagged (Legacy)</dt>
                  <dd>{transaction.isFlaggedFraud === 1 ? "Yes" : "No"}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Created</dt>
                  <dd className="text-xs">{new Date(transaction.createdAt).toLocaleString()}</dd>
                </div>
              </dl>
            </div>
          )}

          {/* Dataset View */}
          {!showOTP && <TransactionDatasetView transaction={transaction} />}

          {/* Actions */}
          {!showOTP && (
            <div className="flex flex-col sm:flex-row gap-3">
              <Button asChild className="flex-1 gap-2">
                <Link to="/simulate">
                  <PlayCircle className="h-4 w-4" aria-hidden="true" />
                  Simulate Another
                </Link>
              </Button>
              <Button asChild variant="outline" className="flex-1 gap-2">
                <Link to="/history">
                  <History className="h-4 w-4" aria-hidden="true" />
                  View History
                </Link>
              </Button>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}
