import { useState } from "react";
import "./App.css";
import { Api, type BannerOffer } from "./Api";

// Initialize API client (point to your backend URL)
const api = new Api({ baseUrl: "http://127.0.0.1:6770" });

function App() {
  const [userId, setUserId] = useState<string>("85686259");
  const [offers, setOffers] = useState<BannerOffer[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setOffers([]);

    try {
      const response = await api.api.getRecommendations({
        user_id: userId,
        limit: 2,
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

  return (
    <div className="app-container">
      <div className="control-panel">
        <div className="input-group">
          <label htmlFor="userId">User ID</label>
          <input
            id="userId"
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Enter User ID (e.g., 85686259)"
          />
        </div>
        <button onClick={handleSearch} disabled={loading || !userId}>
          {loading ? "Calculating..." : "Get Offers"}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="results-area">
        {offers.length > 0 && <h2>Personalized Recommendations</h2>}

        <div className="offers-grid">
          {offers.map((offer) => (
            <div key={offer.product_id} className="offer-card fade-in">
              <div className="offer-icon-wrapper">
                <img
                  src={offer.icon_url}
                  alt={offer.category}
                  className="offer-icon"
                />
              </div>
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

        {!loading && offers.length === 0 && !error && (
          <div className="empty-state">
            Enter a User ID to see predicted Next Best Offers.
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
