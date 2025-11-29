package query

import (
	"context"

	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/domain/offer"
)

type Query struct {
	offerRepo offer.Repository
}

func NewQuery(offerRepo offer.Repository) Query {
	return Query{
		offerRepo: offerRepo,
	}
}

func (q Query) GetRecommendations(ctx context.Context, uid string) ([]offer.Offer, error) {
	return q.offerRepo.GetOffersByUserID(ctx, uid)
}
