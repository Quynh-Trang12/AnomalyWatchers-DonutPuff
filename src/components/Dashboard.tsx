import React, { useState, useEffect } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Cell,
} from "recharts";
import { predictPrimary, TransactionInput } from "../api";
import {
  AlertTriangle,
  ShieldCheck,
  Activity,
  Play,
  Square,
} from "lucide-react";

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState({ safe: 0, fraud: 0, total: 0 });
  const [liveData, setLiveData] = useState<{ time: string; risk: number }[]>(
    [],
  );
  const [isSimulating, setIsSimulating] = useState(false);
  const [currentRisk, setCurrentRisk] = useState(0);

  // Simulation Logic
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSimulating) {
      interval = setInterval(async () => {
        const now = Date.now();
        const isFraudBurst = now % 10000 < 3000;

        let fakeInput: TransactionInput;

        if (isFraudBurst && Math.random() > 0.3) {
          fakeInput = {
            step: 1,
            type: "CASH_OUT",
            amount: 90000 + Math.random() * 50000,
            oldbalanceOrg: 90000 + Math.random() * 50000,
            newbalanceOrig: 0,
            oldbalanceDest: 0,
            newbalanceDest: 0,
          };
        } else {
          fakeInput = {
            step: 1,
            type: "PAYMENT",
            amount: Math.random() * 500,
            oldbalanceOrg: 5000 + Math.random() * 1000,
            newbalanceOrig: 4500 + Math.random() * 1000,
            oldbalanceDest: 0,
            newbalanceDest: 0,
          };
        }

        try {
          const result = await predictPrimary(fakeInput);
          setStats((prev) => ({
            safe: prev.safe + (result.is_fraud ? 0 : 1),
            fraud: prev.fraud + (result.is_fraud ? 1 : 0),
            total: prev.total + 1,
          }));
          setCurrentRisk(result.probability);
          setLiveData((prev) => {
            const newData = [
              ...prev,
              {
                time: new Date().toLocaleTimeString(),
                risk: result.probability,
              },
            ];
            if (newData.length > 30) newData.shift();
            return newData;
          });
        } catch (e) {
          console.error("Simulation error", e);
        }
      }, 800);
    }
    return () => clearInterval(interval);
  }, [isSimulating]);

  const chartData = [
    { name: "Safe", count: stats.safe },
    { name: "Fraud", count: stats.fraud },
  ];

  return (
    <Layout>
      <div className="container py-6 sm:py-8 space-y-8">
        {/* Header Section */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">
              Live Risk Monitor
            </h1>
            <p className="text-muted-foreground mt-1">
              Real-time fraud detection stream powered by XGBoost
            </p>
          </div>

          <div className="flex items-center gap-4 bg-card p-2 rounded-lg border border-border shadow-sm">
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${isSimulating ? "bg-success-muted text-success" : "bg-muted text-muted-foreground"}`}
            >
              {isSimulating ? "System Active" : "System Standby"}
            </span>
            <Button
              onClick={() => setIsSimulating(!isSimulating)}
              variant={isSimulating ? "destructive" : "default"}
              className="gap-2"
            >
              {isSimulating ? (
                <>
                  <Square className="w-4 h-4 fill-current" /> Stop Stream
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" /> Start Stream
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="section-card flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                Transactions Scanned
              </p>
              <h3 className="text-3xl font-bold text-foreground mt-2">
                {stats.total}
              </h3>
            </div>
            <div className="p-3 bg-primary/10 rounded-full">
              <Activity className="w-6 h-6 text-primary" />
            </div>
          </div>

          <div className="section-card flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                Fraud Detected
              </p>
              <h3 className="text-3xl font-bold text-danger mt-2">
                {stats.fraud}
              </h3>
            </div>
            <div className="p-3 bg-danger/10 rounded-full">
              <AlertTriangle className="w-6 h-6 text-danger" />
            </div>
          </div>

          <div className="section-card flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                Current Risk Level
              </p>
              <h3
                className={`text-3xl font-bold mt-2 ${currentRisk > 0.7 ? "text-danger" : currentRisk > 0.3 ? "text-warning" : "text-success"}`}
              >
                {(currentRisk * 100).toFixed(1)}%
              </h3>
            </div>
            <div
              className={`p-3 rounded-full ${currentRisk > 0.7 ? "bg-danger/10" : "bg-success/10"}`}
            >
              <ShieldCheck
                className={`w-6 h-6 ${currentRisk > 0.7 ? "text-danger" : "text-success"}`}
              />
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Live Stream */}
          <div className="section-card">
            <div className="mb-6">
              <h2 className="text-lg font-bold text-foreground">
                Live Fraud Probability
              </h2>
              <p className="text-sm text-muted-foreground">
                Probability stream of incoming transactions
              </p>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer>
                <AreaChart data={liveData}>
                  <defs>
                    <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    stroke="hsl(var(--border))"
                  />
                  <XAxis dataKey="time" hide />
                  <YAxis
                    domain={[0, 1]}
                    tick={{
                      fontSize: 12,
                      fill: "hsl(var(--muted-foreground))",
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: "8px",
                      border: "1px solid hsl(var(--border))",
                      boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                      backgroundColor: "hsl(var(--card))",
                      color: "hsl(var(--card-foreground))",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="risk"
                    stroke="#ef4444"
                    fillOpacity={1}
                    fill="url(#colorRisk)"
                    strokeWidth={2}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Distribution */}
          <div className="section-card">
            <div className="mb-6">
              <h2 className="text-lg font-bold text-foreground">
                Detection Distribution
              </h2>
              <p className="text-sm text-muted-foreground">
                Ratio of Legitimate vs Fraudulent Transactions
              </p>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer>
                <BarChart data={chartData} layout="vertical">
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                    stroke="hsl(var(--border))"
                  />
                  <XAxis type="number" hide />
                  <YAxis
                    dataKey="name"
                    type="category"
                    axisLine={false}
                    tickLine={false}
                    tick={{
                      fontSize: 14,
                      fontWeight: 500,
                      fill: "hsl(var(--foreground))",
                    }}
                  />
                  <Tooltip
                    cursor={{ fill: "transparent" }}
                    contentStyle={{
                      borderRadius: "8px",
                      border: "1px solid hsl(var(--border))",
                      backgroundColor: "hsl(var(--card))",
                    }}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={40}>
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.name === "Fraud" ? "#ef4444" : "#0f766e"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Dashboard;
