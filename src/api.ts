import axios from 'axios';

const API_URL = 'http://localhost:8000';

export interface TransactionInput {
    step: number;
    type: string;
    amount: number;
    oldbalanceOrg: number;
    newbalanceOrig: number;
    oldbalanceDest: number;
    newbalanceDest: number;
}

export interface CreditCardInput {
    amt: number;
    lat: number;
    long: number;
    merch_lat: number;
    merch_long: number;
    dob: string;
    city_pop: number;
}

export interface PredictionOutput {
    probability: number;
    is_fraud: boolean;
    risk_level: 'Low' | 'Medium' | 'High';
    explanation?: string;
}

export const predictPrimary = async (data: TransactionInput): Promise<PredictionOutput> => {
    const response = await axios.post(`${API_URL}/predict/primary`, data);
    return response.data;
};

export const predictSecondary = async (data: CreditCardInput): Promise<PredictionOutput> => {
    const response = await axios.post(`${API_URL}/predict/secondary`, data);
    return response.data;
};
