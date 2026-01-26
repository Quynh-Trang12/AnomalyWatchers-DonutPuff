import { useState, useMemo, useRef, useEffect } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DecisionBadge } from "@/components/ui/DecisionBadge";
import { RiskScore } from "@/components/ui/RiskScore";
import { TransactionDatasetView } from "@/components/result/TransactionDatasetView";
import { Transaction, TransactionType, Decision, TRANSACTION_TYPES } from "@/types/transaction";
import { getTransactions } from "@/lib/storage";
import { Search, Filter, X, ChevronDown, ChevronUp, AlertCircle } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

export default function History() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [typeFilter, setTypeFilter] = useState<TransactionType | "ALL">("ALL");
  const [decisionFilter, setDecisionFilter] = useState<Decision | "ALL">("ALL");
  const [flaggedFilter, setFlaggedFilter] = useState<"ALL" | "YES" | "NO">("ALL");
  const [highRiskOnly, setHighRiskOnly] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    setTransactions(getTransactions().reverse());
    headingRef.current?.focus();
  }, []);

  const filteredTransactions = useMemo(() => {
    return transactions.filter(t => {
      if (typeFilter !== "ALL" && t.type !== typeFilter) return false;
      if (decisionFilter !== "ALL" && t.decision !== decisionFilter) return false;
      if (flaggedFilter === "YES" && t.isFlaggedFraud !== 1) return false;
      if (flaggedFilter === "NO" && t.isFlaggedFraud !== 0) return false;
      if (highRiskOnly && t.riskScore < 0.7) return false;
      return true;
    });
  }, [transactions, typeFilter, decisionFilter, flaggedFilter, highRiskOnly]);

  const clearFilters = () => {
    setTypeFilter("ALL");
    setDecisionFilter("ALL");
    setFlaggedFilter("ALL");
    setHighRiskOnly(false);
  };

  const hasFilters = typeFilter !== "ALL" || decisionFilter !== "ALL" || flaggedFilter !== "ALL" || highRiskOnly;

  return (
    <Layout>
      <div className="container py-6 sm:py-8">
        <div className="max-w-5xl mx-auto">
          <header className="mb-6">
            <h1 
              ref={headingRef}
              tabIndex={-1}
              className="text-2xl sm:text-3xl font-bold mb-2 outline-none"
            >
              Transaction History
            </h1>
            <p className="text-muted-foreground">
              {transactions.length} transaction{transactions.length !== 1 ? "s" : ""} recorded
            </p>
          </header>

          {/* Filters */}
          <div className="section-card mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Filter className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              <span className="font-medium">Filters</span>
              {hasFilters && (
                <Button variant="ghost" size="sm" onClick={clearFilters} className="ml-auto gap-1">
                  <X className="h-3 w-3" aria-hidden="true" />
                  Clear
                </Button>
              )}
            </div>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label htmlFor="type-filter">Type</Label>
                <Select value={typeFilter} onValueChange={(v) => setTypeFilter(v as TransactionType | "ALL")}>
                  <SelectTrigger id="type-filter">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL">All Types</SelectItem>
                    {TRANSACTION_TYPES.map(t => (
                      <SelectItem key={t.value} value={t.value}>{t.value}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="decision-filter">Decision</Label>
                <Select value={decisionFilter} onValueChange={(v) => setDecisionFilter(v as Decision | "ALL")}>
                  <SelectTrigger id="decision-filter">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL">All Decisions</SelectItem>
                    <SelectItem value="APPROVE">Approved</SelectItem>
                    <SelectItem value="STEP_UP">Step-Up</SelectItem>
                    <SelectItem value="BLOCK">Blocked</SelectItem>
                    <SelectItem value="APPROVE_AFTER_STEPUP">Approved (After Step-Up)</SelectItem>
                    <SelectItem value="BLOCK_STEPUP_FAILED">Blocked (Step-Up Failed)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="flagged-filter">Legacy Flagged</Label>
                <Select value={flaggedFilter} onValueChange={(v) => setFlaggedFilter(v as "ALL" | "YES" | "NO")}>
                  <SelectTrigger id="flagged-filter">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ALL">All</SelectItem>
                    <SelectItem value="YES">Flagged (isFlaggedFraud=1)</SelectItem>
                    <SelectItem value="NO">Not Flagged</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="high-risk">High Risk Only</Label>
                <Button
                  id="high-risk"
                  variant={highRiskOnly ? "default" : "outline"}
                  className="w-full justify-start"
                  onClick={() => setHighRiskOnly(!highRiskOnly)}
                  aria-pressed={highRiskOnly}
                >
                  {highRiskOnly ? "✓ Risk ≥ 70%" : "Risk ≥ 70%"}
                </Button>
              </div>
            </div>
          </div>

          {/* Results */}
          {filteredTransactions.length === 0 ? (
            <div className="section-card text-center py-12">
              <p className="text-muted-foreground">
                {transactions.length === 0
                  ? "No transactions yet. Start by creating a simulation."
                  : "No transactions match the selected filters."}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredTransactions.map(transaction => (
                <button
                  key={transaction.id}
                  onClick={() => setSelectedTransaction(transaction)}
                  className="w-full text-left section-card hover:shadow-md transition-shadow focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                  aria-label={`View details for ${transaction.type} transaction of ${transaction.amount.toLocaleString()}`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold">{transaction.type}</span>
                        <span className="text-muted-foreground">•</span>
                        <span className="font-mono text-sm">{transaction.amount.toLocaleString()}</span>
                        {transaction.isFlaggedFraud === 1 && (
                          <span className="text-xs bg-warning-muted text-warning px-2 py-0.5 rounded">
                            Flagged
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-muted-foreground truncate">
                        <span className="font-mono">{transaction.nameOrig}</span>
                        <span className="mx-2">→</span>
                        <span className="font-mono">{transaction.nameDest}</span>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        Step {transaction.step} • {new Date(transaction.createdAt).toLocaleString()}
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="w-24">
                        <RiskScore score={transaction.riskScore} showBar={false} size="sm" />
                      </div>
                      <DecisionBadge decision={transaction.decision} size="sm" />
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Transaction Detail Dialog */}
      <Dialog open={!!selectedTransaction} onOpenChange={() => setSelectedTransaction(null)}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Transaction Details</DialogTitle>
          </DialogHeader>
          
          {selectedTransaction && (
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <DecisionBadge decision={selectedTransaction.decision} size="md" />
                <RiskScore score={selectedTransaction.riskScore} showBar={false} size="md" />
              </div>

              <div className="section-card">
                <h3 className="font-semibold mb-3">Summary</h3>
                <dl className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <dt className="text-muted-foreground">Type</dt>
                    <dd className="font-medium">{selectedTransaction.type}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Amount</dt>
                    <dd className="font-mono font-medium">{selectedTransaction.amount.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Origin</dt>
                    <dd className="font-mono text-xs">{selectedTransaction.nameOrig}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Destination</dt>
                    <dd className="font-mono text-xs">{selectedTransaction.nameDest}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Flagged (Legacy)</dt>
                    <dd>{selectedTransaction.isFlaggedFraud === 1 ? "Yes" : "No"}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Labeled as Fraud</dt>
                    <dd>{selectedTransaction.isFraud === 1 ? "Yes" : "No"}</dd>
                  </div>
                </dl>
              </div>

              <div className="section-card">
                <h3 className="font-semibold mb-3">Risk Factors</h3>
                <ul className="space-y-2">
                  {selectedTransaction.reasons.map((reason, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <AlertCircle className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" aria-hidden="true" />
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>

              <TransactionDatasetView transaction={selectedTransaction} defaultOpen />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </Layout>
  );
}
