package main

import (
	"log"
	"os"

	"github.com/go-openapi/loads"
	flags "github.com/jessevdk/go-flags"

	"github.com/AlmostLate/systemd-timesyncd/src/backend/generated/restapi"
	"github.com/AlmostLate/systemd-timesyncd/src/backend/generated/restapi/operations"
	application "github.com/AlmostLate/systemd-timesyncd/src/backend/internal/app"
	"github.com/AlmostLate/systemd-timesyncd/src/backend/internal/ports"
)

func main() {
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

	parser := flags.NewParser(server, flags.Default)
	parser.ShortDescription = "PSB Recommendation Engine"
	parser.LongDescription = "API for retrieving personalized banking product recommendations."

	server.ConfigureFlags()
	for _, optsGroup := range api.CommandLineOptionsGroups {
		_, err := parser.AddGroup(optsGroup.ShortDescription, optsGroup.LongDescription, optsGroup.Options)
		if err != nil {
			log.Fatalln(err)
		}
	}

	if _, err := parser.Parse(); err != nil {
		code := 1
		if fe, ok := err.(*flags.Error); ok {
			if fe.Type == flags.ErrHelp {
				code = 0
			}
		}
		os.Exit(code)
	}

	server.ConfigureAPI()

	log.Printf("Starting server on port %d...", server.Port)
	if err := server.Serve(); err != nil {
		log.Fatalln(err)
	}
}
