Feature: Backend emit is pure, deterministic, and target-idiomatic
  The OSS-able unit of tile-rs is the pure emit:
  convert_mlir_to_<target>(mlir_text) -> Result<String, String>, with NO
  LLVM/CANN toolchain present. For the open Metal backend, emitting a softmax
  kernel must be a pure function of its input (stable across runs) and must
  contain the target's signature idiom.

  Background:
    Given the canonical softmax MLIR snippet

  Scenario Outline: <target> emits idiomatic source deterministically
    When I emit the snippet to target "<target>"
    Then the emit succeeds
    And the output contains "<idiom>"
    And emitting the same snippet again yields byte-identical output

    Examples:
      | target  | idiom            |
      | msl     | kernel void      |

  Scenario: an empty MLIR module is rejected, not silently emitted
    When I emit an empty module to target "msl"
    Then the emit fails
