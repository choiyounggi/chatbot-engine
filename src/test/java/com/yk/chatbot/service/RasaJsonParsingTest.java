package com.yk.chatbot.service;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Covers RasaClientService#parseResponse directly against the Jackson 3
 * (tools.jackson.databind) response-parsing path: normal parsing, a
 * container-valued entity (which Jackson 3's asString() without a default
 * would throw JsonNodeException on, unlike Jackson 2's asText()), the
 * empty-object boundary, and a non-array entities field.
 */
class RasaJsonParsingTest {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final RasaClientService service = new RasaClientService(objectMapper);

    @Test
    void parsesIntentConfidenceAndStringEntities() {
        JsonNode response = objectMapper.readTree(
                "{\"intent\":{\"name\":\"greet\",\"confidence\":0.95},"
                        + "\"entities\":[{\"entity\":\"city\",\"value\":\"Seoul\"},"
                        + "{\"entity\":\"day\",\"value\":\"Monday\"}]}");

        Map<String, Object> result = service.parseResponse(response);

        assertThat(result.get("intent")).isEqualTo("greet");
        assertThat(result.get("confidence")).isEqualTo(0.95);
        @SuppressWarnings("unchecked")
        Map<String, Object> entities = (Map<String, Object>) result.get("entities");
        assertThat(entities).containsOnly(Map.entry("city", "Seoul"), Map.entry("day", "Monday"));
    }

    @Test
    void objectValuedEntityIsSkippedButIntentAndStringEntityStillParse() {
        JsonNode response = objectMapper.readTree(
                "{\"intent\":{\"name\":\"greet\",\"confidence\":0.9},"
                        + "\"entities\":[{\"entity\":\"range\",\"value\":{\"from\":\"a\",\"to\":\"b\"}},"
                        + "{\"entity\":\"city\",\"value\":\"Seoul\"}]}");

        Map<String, Object> result = service.parseResponse(response);

        assertThat(result.get("intent")).isEqualTo("greet");
        @SuppressWarnings("unchecked")
        Map<String, Object> entities = (Map<String, Object>) result.get("entities");
        assertThat(entities).containsOnly(Map.entry("city", "Seoul"));
    }

    @Test
    void emptyJsonObjectYieldsUnknownIntentZeroConfidenceAndNoEntities() {
        JsonNode response = objectMapper.readTree("{}");

        Map<String, Object> result = service.parseResponse(response);

        assertThat(result.get("intent")).isEqualTo("unknown");
        assertThat(result.get("confidence")).isEqualTo(0.0);
        @SuppressWarnings("unchecked")
        Map<String, Object> entities = (Map<String, Object>) result.get("entities");
        assertThat(entities).isEmpty();
    }

    @Test
    void nonArrayEntitiesFieldYieldsEmptyEntitiesWithoutThrowing() {
        JsonNode response = objectMapper.readTree(
                "{\"intent\":{\"name\":\"greet\",\"confidence\":0.9},\"entities\":\"not-an-array\"}");

        Map<String, Object> result = service.parseResponse(response);

        assertThat(result.get("intent")).isEqualTo("greet");
        @SuppressWarnings("unchecked")
        Map<String, Object> entities = (Map<String, Object>) result.get("entities");
        assertThat(entities).isEmpty();
    }
}
