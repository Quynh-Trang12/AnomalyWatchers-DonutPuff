from pydantic import BaseModel
from typing import Optional

class TransactionInput(BaseModel):
    step: int = 1
    type: str  # CASH_OUT, PAYMENT, etc.
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

class CreditCardInput(BaseModel):
    amt: float
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    dob: str # YYYY-MM-DD
    city_pop: int

class PredictionOutput(BaseModel):
    probability: float
    is_fraud: bool
    risk_level: str # Low, Medium, High
    explanation: Optional[str] = None
