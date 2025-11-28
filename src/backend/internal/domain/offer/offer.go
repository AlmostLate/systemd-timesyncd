package offer

import "context"

type Offer struct {
	productID   string
	title       string
	description string
	icon        string
	category    string
}

func New(id, title, desc, icon, cat string) Offer {
	return Offer{
		productID:   id,
		title:       title,
		description: desc,
		icon:        icon,
		category:    cat,
	}
}

func (of Offer) GetProductID() string {
	return of.productID
}

func (of Offer) GetTitle() string {
	return of.title
}

func (of Offer) GetDescription() string {
	return of.description
}

func (of Offer) GetIcon() string {
	return of.icon
}

func (of Offer) GetCategory() string {
	return of.category
}

type Repository interface {
	GetOffersByUserID(ctx context.Context, userID string) ([]Offer, error)
}
