package valkey

import (
	"context"

	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/domain/offer"
)

type MockRepository struct{}

func NewMockRepository() *MockRepository {
	return &MockRepository{}
}

func (r *MockRepository) GetOffersByUserID(ctx context.Context, userID string) ([]offer.Offer, error) {
	// Key scheme: "nbo:{user_id}"

	// Mock logic: return specific offers for a demo user, empty for others, or default for generic
	if userID == "demo_user" {
		return []offer.Offer{
			offer.New(
				"card_credit_100days",
				"100 Days without %",
				"The most popular credit card with a long grace period.",
				"https://cdn.psbank.ru/icons/cr_100.png",
				"credit_cards",
			),
			offer.New(
				"loan_cash_premium",
				"Cash Loan Premium",
				"Pre-approved 1,000,000 RUB with 5.9% rate.",
				"https://cdn.psbank.ru/icons/loan_p.png",
				"loans",
			),
		}, nil
	}

	// Default fallback for the mock
	return []offer.Offer{
		offer.New(
			"debit_all_inclusive",
			"All Inclusive Debit",
			"Cashback on everything.",
			"https://cdn.psbank.ru/icons/deb_ai.png",
			"cards",
		),
	}, nil
}
