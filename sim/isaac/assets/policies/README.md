Place trained Go2 locomotion policies here for `--locomotion-mode rl`.

Expected runtime path example:

```powershell
.\sim\run_sim.bat --locomotion-mode rl --rl-policy-path sim\isaac\assets\policies\go2_policy.pt
```

The policy file must be local. Do not load policies directly from remote URLs or Nucleus paths at simulation runtime.
