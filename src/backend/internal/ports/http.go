package ports

import (
	"github.com/go-openapi/runtime/middleware"

	"github.com/AlmostLate/systemd-timesyncd/src/backend/generated/models"
	"github.com/AlmostLate/systemd-timesyncd/src/backend/generated/restapi/operations"
	application "github.com/AlmostLate/systemd-timesyncd/src/backend/internal/app"
)

type HTTPServer struct {
	app application.Application
}

func NewHTTPServer(app application.Application) *HTTPServer {
	return &HTTPServer{
		app: app,
	}
}

func (h *HTTPServer) Handle(params operations.GetRecommendationsParams) middleware.Responder {
	ctx := params.HTTPRequest.Context()
	userID := params.UserID

	offers, err := h.app.Queries.GetRecommendations(ctx, userID)
	if err != nil {
		return operations.NewGetRecommendationsInternalServerError().WithPayload(&models.Error{
			Code:    500,
			Message: "Internal system error",
		})
	}

	payloadOffers := make([]*models.BannerOffer, 0, len(offers))

	for _, o := range offers {
		mappedOffer := &models.BannerOffer{
			ProductID:   o.GetProductID(),
			Title:       o.GetTitle(),
			Description: o.GetDescription(),
			Category:    o.GetCategory(),
		}
		payloadOffers = append(payloadOffers, mappedOffer)
	}

	return operations.NewGetRecommendationsOK().WithPayload(&models.RecommendationResponse{
		UserID: userID,
		Offers: payloadOffers,
	})
}
