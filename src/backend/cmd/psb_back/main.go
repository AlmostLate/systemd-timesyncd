package main

import (
	"errors"
	"log"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/go-openapi/loads"
	flags "github.com/jessevdk/go-flags"
	"github.com/rs/cors"

	"github.com/AlmostLate/systemd-timesyncd/src/backend/generated/restapi"
	"github.com/AlmostLate/systemd-timesyncd/src/backend/generated/restapi/operations"
	application "github.com/AlmostLate/systemd-timesyncd/src/backend/internal/app"
	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/ports"
)

type options struct {
	CORSAllowedOrigins []string `long:"cors-allowed-origins" description:"A list of allowed origins for CORS" required:"false"`
}

const (
	envKeyCORS = `CORS_ALLOWED_ORIGINS`
)

func main() {
	var opts options

	swaggerSpec, err := loads.Embedded(restapi.SwaggerJSON, restapi.FlatSwaggerJSON)
	if err != nil {
		log.Fatalln(err)
	}

	api := operations.NewPSBRecommendationEngineAPIAPI(swaggerSpec)

	app := application.NewApplication()

	httpHandler := ports.NewHTTPServer(app)

	api.GetRecommendationsHandler = operations.GetRecommendationsHandlerFunc(httpHandler.Handle)

	server := restapi.NewServer(api)
	defer server.Shutdown()

	parseFlags(server, api, &opts)

	server.ConfigureAPI()

	allowedOrigins := opts.CORSAllowedOrigins
	if envCors, present := os.LookupEnv(envKeyCORS); len(opts.CORSAllowedOrigins) <= 0 && present {
		allowedOrigins = strings.Split(envCors, ",")
	}

	if len(allowedOrigins) == 0 {
		log.Println("CORS allowed origins must be provided via --cors-allowed-origins flag or CORS_ALLOWED_ORIGINS env var")

		return
	}

	server.SetHandler(applyCors(allowedOrigins, api.Serve(nil)))

	log.Printf("Starting server on port %d...", server.Port)
	if err := server.Serve(); err != nil {
		log.Fatalln(err)
	}
}

func applyCors(allowedOrigins []string, handler http.Handler) http.Handler {
	corsOptions := cors.New(cors.Options{
		AllowedOrigins:   allowedOrigins,
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type"},
		AllowCredentials: true,
	})

	handler = corsOptions.Handler(handler)

	return handler
}

func parseFlags(server *restapi.Server, api *operations.PSBRecommendationEngineAPIAPI, opts *options) {
	parser := flags.NewParser(server, flags.Default)

	_, err := parser.AddGroup("Application Options", "Options for the application", opts)
	if err != nil {
		log.Printf("parser add group %s", err)
		os.Exit(1)
	}

	server.ConfigureFlags()

	for _, optsGroup := range api.CommandLineOptionsGroups {
		_, err = parser.AddGroup(optsGroup.ShortDescription, optsGroup.LongDescription, optsGroup.Options)
		if err != nil {
			log.Printf("parser add group %s", err)
			os.Exit(1)
		}
	}

	if _, err = parser.Parse(); err != nil {
		flagErr := &flags.Error{}
		if errors.As(err, &flagErr) && flagErr.Type != flags.ErrHelp {
			slog.Error("parse flags", slog.Any("err", flagErr))
			os.Exit(1)
		}
	}
}
