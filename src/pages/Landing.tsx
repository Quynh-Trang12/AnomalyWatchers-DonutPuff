import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Layout } from "@/components/layout/Layout";
import { 
  Shield, 
  PlayCircle, 
  History, 
  Settings,
  Database,
  ArrowRight,
  CheckCircle
} from "lucide-react";

const datasetColumns = [
  "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
  "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"
];

const features = [
  {
    icon: PlayCircle,
    title: "Transaction Simulator",
    description: "Create synthetic transactions matching the Kaggle dataset structure",
  },
  {
    icon: Shield,
    title: "Risk Scoring Engine",
    description: "Rule-based fraud detection model inspired by dataset patterns",
  },
  {
    icon: Database,
    title: "Persistent Storage",
    description: "All transactions and settings saved locally for demo continuity",
  },
];

export default function Landing() {
  return (
    <Layout>
      {/* Hero Section */}
      <section className="bg-gradient-hero py-12 sm:py-16 lg:py-20">
        <div className="container">
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
              <Shield className="h-4 w-4" aria-hidden="true" />
              Educational Demo
            </div>
            
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
              Online Payments Fraud Detection
              <span className="block text-primary">Simulator</span>
            </h1>
            
            <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
              Explore fraud detection concepts using the Kaggle "Online Payments Fraud Detection" dataset. 
              Create transactions, analyze risk scores, and understand how detection systems work.
            </p>

            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Button asChild size="lg" className="gap-2">
                <Link to="/simulate">
                  <PlayCircle className="h-5 w-5" aria-hidden="true" />
                  Start Simulation
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="gap-2">
                <Link to="/history">
                  <History className="h-5 w-5" aria-hidden="true" />
                  View History
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="gap-2">
                <Link to="/admin">
                  <Settings className="h-5 w-5" aria-hidden="true" />
                  Admin Console
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Dataset Columns */}
      <section className="py-12 border-b border-border">
        <div className="container">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-xl font-semibold text-center mb-6">
              Dataset Columns Used
            </h2>
            <div className="flex flex-wrap justify-center gap-2">
              {datasetColumns.map(col => (
                <span
                  key={col}
                  className="px-3 py-1.5 bg-muted rounded-md font-mono text-sm"
                >
                  {col}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-12 sm:py-16">
        <div className="container">
          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {features.map(feature => (
              <div key={feature.title} className="section-card text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 mb-4">
                  <feature.icon className="h-6 w-6 text-primary" aria-hidden="true" />
                </div>
                <h3 className="font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section className="py-12 sm:py-16 bg-muted/30">
        <div className="container">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-2xl font-semibold text-center mb-8">How It Works</h2>
            
            <div className="space-y-4">
              {[
                "Enter transaction details matching the Kaggle dataset fields",
                "The mock model computes a risk score using rule-based features",
                "Transactions are approved, flagged for step-up, or blocked",
                "Review decisions and explore the scoring logic",
                "Customize thresholds and rules in the Admin console",
              ].map((step, index) => (
                <div key={index} className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-medium">
                    {index + 1}
                  </div>
                  <p className="pt-1 text-muted-foreground">{step}</p>
                </div>
              ))}
            </div>

            <div className="mt-8 text-center">
              <Button asChild size="lg" className="gap-2">
                <Link to="/simulate">
                  Get Started
                  <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </Layout>
  );
}
