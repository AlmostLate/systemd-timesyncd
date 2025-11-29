interface AppConfig {
  API_BASE_URL: string;
  KIOSK_UIDS: Array<string>;
}

interface Window {
  config: AppConfig;
}
