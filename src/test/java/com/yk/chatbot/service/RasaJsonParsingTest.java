package com.yk.chatbot.service;

import org.junit.jupiter.api.Test;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Jackson 3 (tools.jackson.databind) migration coverage for the
 * response-parsing pattern used by RasaClientService#parseMessage and
 * WeatherService#getWeatherForLocation (readTree + has()/get() navigation).
 */
class RasaJsonParsingTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void parsesIntentAndEntitiesFromValidRasaResponse() {
        String json = "{\"intent\":{\"name\":\"greet\",\"confidence\":0.95},"
                + "\"entities\":[{\"entity\":\"city\",\"value\":\"Seoul\"}]}";

        JsonNode response = objectMapper.readTree(json);

        assertThat(response.has("intent")).isTrue();
        assertThat(response.get("intent").get("name").asString()).isEqualTo("greet");
        assertThat(response.get("intent").get("confidence").asDouble()).isEqualTo(0.95);
        assertThat(response.get("entities").get(0).get("value").asString()).isEqualTo("Seoul");
    }

    @Test
    void readTreeThrowsJacksonExceptionOnMalformedJson() {
        String malformed = "{\"intent\": {\"name\": \"greet\"";

        assertThrows(JacksonException.class, () -> objectMapper.readTree(malformed));
    }

    @Test
    void emptyJsonObjectHasNoIntentFieldAndIsEmpty() {
        JsonNode response = objectMapper.readTree("{}");

        assertThat(response.isEmpty()).isTrue();
        assertThat(response.has("intent")).isFalse();
    }
}
