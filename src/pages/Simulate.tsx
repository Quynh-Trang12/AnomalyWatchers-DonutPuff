import { Layout } from "@/components/layout/Layout";
import { TransactionForm } from "@/components/simulator/TransactionForm";

export default function Simulate() {
  return (
    <Layout>
      <div className="container py-6 sm:py-8 pb-28">
        <div className="max-w-5xl mx-auto">
          <header className="mb-6 sm:mb-8">
            <h1 className="text-2xl sm:text-3xl font-bold mb-2">
              Transaction Simulator
            </h1>
            <p className="text-muted-foreground">
              Create a synthetic transaction using the Kaggle dataset field names. 
              The mock model will compute a risk score and decision.
            </p>
          </header>

          <TransactionForm />
        </div>
      </div>
    </Layout>
  );
}
