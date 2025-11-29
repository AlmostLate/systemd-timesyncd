package application

import (
	"database/sql"

	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/app/query"
	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/repository/chdb"
)

type Application struct {
	Queries query.Query
}

func NewApplication(db *sql.DB) Application {
	offerRepo := chdb.NewRepository(db)

	return Application{
		Queries: query.NewQuery(offerRepo),
	}
}
