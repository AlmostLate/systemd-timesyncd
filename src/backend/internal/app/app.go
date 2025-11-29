package application

import (
	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/app/query"
	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/repository/valkey"
)

type Application struct {
	Queries query.Query
}

func NewApplication() Application {
	offerRepo := valkey.NewMockRepository()

	return Application{
		Queries: query.NewQuery(offerRepo),
	}
}
