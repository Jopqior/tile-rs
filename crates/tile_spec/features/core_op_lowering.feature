Feature: The open Metal backend lowers the core tile-op set
  tile-rs claims generality across the accelerator long tail: the same core
  ops (rms_norm, matmul, softmax, rope, elementwise add) must lower through
  every backend. The Metal emitter is the arm published as source, so this
  pins one scenario per op against it. Closed backends carry the same claim in
  features/closed/.

  Scenario Outline: the <op> op lowers to <target>
    Given the canonical "<op>" MLIR snippet
    When I emit the snippet to target "<target>"
    Then the emit succeeds
    And the output contains "<evidence>"
    And emitting the same snippet again yields byte-identical output

    Examples: core ops
      | op       | target | evidence    |
      | rms_norm | msl    | kernel void |
      | matmul   | msl    | kernel void |
      | softmax  | msl    | kernel void |
      | rope     | msl    | kernel void |
      | add      | msl    | kernel void |
