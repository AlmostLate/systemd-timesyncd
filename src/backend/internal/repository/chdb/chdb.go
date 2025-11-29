package chdb

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"

	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/domain/offer"
)

// Repository implements the offer.Repository interface using a ClickHouse database.
type Repository struct {
	conn *sql.DB
}

// NewRepository creates a new instance of the ClickHouse offer repository.
func NewRepository(db *sql.DB) *Repository {
	return &Repository{conn: db}
}

// GetOffersByUserID retrieves personalized offers for a given user from ClickHouse.
// It queries the recommendations table for a user's ranked offer IDs and joins
// this data with the offers table to get the complete offer details.
func (r *Repository) GetOffersByUserID(ctx context.Context, userID string) ([]offer.Offer, error) {
	uid, err := strconv.ParseUint(userID, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid user ID format: %w", err)
	}

	const query = `
		SELECT
			o.product_id,
			o.product_name,
			o.description,
			o.product_type
		FROM recommendations AS r
		INNER JOIN offers AS o ON r.offer_id = o.product_id
		WHERE r.user_id = ?
		ORDER BY r.score DESC
		LIMIT 10
	`

	rows, err := r.conn.QueryContext(ctx, query, uid)
	if err != nil {
		return nil, fmt.Errorf("failed to query offers: %w", err)
	}
	defer rows.Close()

	var offers []offer.Offer
	for rows.Next() {
		var (
			productID   string
			title       string
			description string
			category    string
		)

		if err = rows.Scan(&productID, &title, &description, &category); err != nil {
			return nil, fmt.Errorf("failed to scan offer row: %w", err)
		}

		// The icon field is intentionally left empty as per the requirements.
		offers = append(offers, offer.New(productID, title, description, "", category))
	}

	if err = rows.Err(); err != nil {
		return nil, fmt.Errorf("error during row iteration: %w", err)
	}

	return offers, nil
}
