from dataclasses import dataclass
from collections import defaultdict

@dataclass
class UsedOffer:
    offer_id: int
    ts: int

@dataclass
class User:
    uid: int
    socdem_cluster: float
    region: float
    
    used_offers: list[UsedOffer]



@dataclass
class LeadUser:
    uid: int
    socdem_cluster: float
    region: float

@dataclass
class Offer:
    offer_id: int
    score: float

def calculate_score_for_user(user: User, top_lead_users: list[User]) -> list[Offer]:
    top_users_count = len(top_lead_users)

    user_used_offers = sorted(user.used_offers, key=lambda uo: uo.ts)
    user_last_offer_ts = user_used_offers[-1].ts

    user_used_offers_ids = {uo.offer_id for uo in user_used_offers}

    offer_scores = defaultdict(float)
    for lead_user in top_lead_users:
        lead_offers = [lo for lo in lead_user.used_offers if lo.ts > user_last_offer_ts and lo.offer_id not in user_used_offers_ids]
        lead_offers = sorted(lead_offers, key=lambda lo: lo.ts)

        closest_offerst_checked = 0

        check_max = 5
        for lead_offer in lead_offers:
            if closest_offerst_checked == check_max:
                break
            position_weight = (check_max - closest_offerst_checked) / check_max
            region_bonus = 0.5 if lead_user.region == user.region else 0.0
            socdem_bonus = 0.5 if lead_user.socdem_cluster == user.socdem_cluster else 0.0
            offer_scores[lead_offer.offer_id] += position_weight * (1.0 + region_bonus + socdem_bonus) / top_users_count
            closest_offerst_checked += 1
    
    offers = []
    for offer in sorted(offer_scores.items(), key=lambda o: o[1], reverse=True):
        offers.append(Offer(offer[0], offer[1]))
    
    return offers

