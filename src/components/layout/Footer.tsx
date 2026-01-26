import { Shield } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border bg-muted/30 mt-auto">
      <div className="container py-6 text-center text-sm text-muted-foreground">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Shield className="h-4 w-4" aria-hidden="true" />
          <span className="font-medium">Online Payments Fraud Detection — Simulator</span>
        </div>
        <p className="max-w-md mx-auto">
          Educational demo based on the Kaggle dataset. 
          No real transactions or financial operations occur.
        </p>
        <p className="mt-2 text-xs text-muted-foreground/70">
          Dataset columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, 
          nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
        </p>
      </div>
    </footer>
  );
}
