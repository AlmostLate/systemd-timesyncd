import { useState, useEffect } from "react";
import "./App.css";
import { Api, type BannerOffer } from "./Api";

const KIOSK_UIDS = window.config.KIOSK_UIDS;
const API_BASE_URL = window.config.API_BASE_URL;

const api = new Api({ baseUrl: API_BASE_URL });

function App() {
  const [selectedUserId, setSelectedUserId] = useState<string>(KIOSK_UIDS[0]);
  const [offers, setOffers] = useState<BannerOffer[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedUserId) {
      setOffers([]);
      return;
    }

    const fetchRecommendations = async () => {
      setLoading(true);
      setError(null);
      setOffers([]);

      try {
        const response = await api.api.getRecommendations({
          user_id: selectedUserId,
          limit: 3,
        });
        setOffers(response.data.offers);
      } catch (err) {
        console.error(err);
        setError(
          "Failed to load recommendations. Please check the backend connection."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [selectedUserId]);

  return (
    <div className="app-container">
      <div className="header">
        <h1>Next Best Offer</h1>
      </div>

      <div className="control-panel">
        <div className="input-group">
          <label htmlFor="user-select">Select User</label>
          <select
            id="user-select"
            value={selectedUserId}
            onChange={(e) => setSelectedUserId(e.target.value)}
            disabled={KIOSK_UIDS.length === 0}
          >
            {KIOSK_UIDS.map((uid) => (
              <option key={uid} value={uid}>
                {uid}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="results-area">
        {loading && <div className="loading-state">Calculating...</div>}

        {!loading && !error && offers.length > 0 && (
          <div className="offers-grid">
            {offers.map((offer) => (
              <div key={offer.product_id} className="offer-card fade-in">
                <div className="offer-content">
                  <span className="offer-category">
                    {offer.category.toUpperCase()}
                  </span>
                  <h3>{offer.title}</h3>
                  <p>{offer.description}</p>
                  <button className="cta-button">Оформить</button>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && !error && offers.length === 0 && (
          <div className="empty-state">
            {selectedUserId
              ? "No recommendations available for this user."
              : "Please select a user to see recommendations."}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
