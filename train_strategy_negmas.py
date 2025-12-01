# train_strategy_negmas.py
# THE GYM: Offline Simulation (Fixed for realistic deal flow)

import random
import statistics
import numpy as np
import nashpy as nash

def format_inr(x): return f"₹{x:,}"

# 1. SETUP SIMULATION ENVIRONMENT
ITEM_NAME = "50 MacBooks"
MV = 50_00_000  # 50 Lakhs
TARGET = int(MV * 0.60)
MAX_BUDGET = int(MV * 0.90)
# INCREASED ROUNDS: Real deals take time. 
# Giving them 15 rounds allows the "Soft" vs "Hard" difference to actually play out.
MAX_ROUNDS = 15 

# 2. DEFINING THE "GAME"

def get_buyer_move(stance, current_offer, seller_price):
    if current_offer == 0: return TARGET
    
    gap = seller_price - current_offer
    if gap <= 0: return current_offer
    
    # Tunable Parameters
    if stance == "HARD": concession = 0.05
    elif stance == "MEDIUM": concession = 0.15
    else: concession = 0.35
    
    step = max(int(gap * concession), 5000)
    new = current_offer + step
    return min(new, MAX_BUDGET)

def get_seller_move(style, current_price, buyer_offer, min_price):
    gap = current_price - buyer_offer
    if gap <= 0: return current_price
    
    if style == "stubborn": drop = 0.05
    elif style == "flexible": drop = 0.25
    else: drop = 0.10
    
    new = current_price - int(gap * drop)
    return max(new, min_price)

# 3. RUNNING THE GYM
def run_gym():
    print("--- NEGOTIATION GYM STARTING (Calibrated) ---")
    
    results = {}
    
    for stance in ["HARD", "MEDIUM", "SOFT"]:
        wins = 0
        total_savings = 0
        
        for _ in range(500): 
            # Randomize Seller
            seller_style = random.choice(["stubborn", "flexible", "normal"])
            # Seller Floor: 75% to 88% of MV (Realistic margins)
            seller_min = int(MV * random.uniform(0.75, 0.88))
            # Seller Ask: 110% to 130% of MV (Not insane 160% pricing)
            seller_ask = int(MV * random.uniform(1.1, 1.3))
            
            buyer_offer = 0
            rounds = 0
            deal = False
            
            while rounds < MAX_ROUNDS:
                rounds += 1
                
                # Buyer Move
                buyer_offer = get_buyer_move(stance, buyer_offer, seller_ask)
                
                # Check Deal (Buyer met Seller)
                if buyer_offer >= seller_ask: 
                    deal = True; break
                
                # Seller Move
                seller_ask = get_seller_move(seller_style, seller_ask, buyer_offer, seller_min)
                
                # Check Deal (Seller met Buyer)
                if seller_ask <= buyer_offer:
                    deal = True; break
            
            if deal:
                wins += 1
                # Savings = Market Value - Final Price
                # If we paid less than MV, that's savings.
                final_price = max(buyer_offer, seller_ask) # roughly where they met
                total_savings += (MV - final_price)
        
        # Calculate Stats
        win_rate = (wins / 500) * 100
        avg_sav = total_savings / 500 # Spread across all deals (including failures as 0 savings)
        
        results[stance] = {"win_rate": win_rate, "avg_savings": avg_sav}
        print(f"Stance {stance:6}: Win Rate {win_rate:5.1f}% | Avg Savings {format_inr(int(avg_sav))}")

    print("\n--- OPTIMIZATION RESULT ---")
    # We pick the strategy that maximizes SAVINGS (not just wins)
    best_stance = max(results, key=lambda k: results[k]['avg_savings'])
    print(f"RECOMMENDED DEFAULT STANCE: {best_stance}")
    print("Use this to set the default in decide_stance_with_nash() in negotiator.py")

if __name__ == "__main__":
    run_gym()