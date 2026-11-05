# AETHRION capability architecture

## Goal

AETHRION presents one product interface while dispatching explicitly selected
work to specialist backends. The base AETHRIONLM is a decoder-only text model;
it is not, by itself, an image generator, 3D generator, speech system, or
repository-editing agent.

## Capability map

| UI selection | Capability key | Intended backend |
|---|---|---|
| Chat | chat | AETHRION text model |
| Code | code | Code model plus a sandboxed coding agent |
| Image | image | Image-generation model/service |
| 3D | 3d | Text/image-to-3D model/service |
| Research | research | Search, retrieval, and source-verification tools |
| File analysis | file_analysis | Format-aware parsers and analysis models |
| Audio | audio | Speech recognition and synthesis services |

## Dispatch contract

The aethrion.routing.CapabilityRouter module is the first backend-neutral
dispatch boundary. The UI or API must pass the selected capability explicitly.
A provider is available only after its real handler is registered. An
unconfigured capability raises CapabilityUnavailable; it must not silently
fall back to text chat or return a fabricated success.

Provider handlers receive a CapabilityRequest and return their real result.
Credentials belong in environment variables or a secret manager, never in
source files, committed configuration, logs, or client bundles. Provider
adapters must validate input sizes, timeouts, response types, and output
artifacts. File-writing or repository-editing agents must use an isolated
workspace, inspect diffs, run tests, and require approval before publishing
changes.

## Delivery phases

1. Keep the text-model training pipeline independent and fail-closed.
2. Add and test the capability registry/router.
3. Implement one provider adapter per capability, starting with code and image.
4. Add the 3D provider adapter and validate exported GLB/GLTF/OBJ artifacts.
5. Connect the product UI to the API's explicit capability key.
6. Add integration tests with mocked providers, then opt-in live provider tests.
7. Measure each capability separately; do not claim parity with other assistants
   without reproducible benchmark results.

## Current implementation status

The routing contract and unit tests are present. This is infrastructure, not a
claim that image, 3D, audio, research, or code providers are already connected.
Those capabilities remain unavailable until their adapters and credentials are
configured and integration-tested.
