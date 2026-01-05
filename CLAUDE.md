## Version Control: TL Client (Timelapse)

**USE `./tl-client.sh` FOR ALL VERSION CONTROL - NOT GIT**

TL (Timelapse) replaces git for checkpoint management. It provides automatic, instant snapshots of the working directory.

### Commands Reference

| Command | Purpose |
|---------|---------|
| `./tl-client.sh setup` | Initialize + start daemon (run first in any repo) |
| `./tl-client.sh save` | Force immediate checkpoint |
| `./tl-client.sh log [n]` | View checkpoint history |
| `./tl-client.sh restore <id>` | Restore to checkpoint (overwrites current!) |
| `./tl-client.sh diff <a> <b>` | Compare two checkpoints |
| `./tl-client.sh pin <id> <name>` | Name important checkpoints |
| `./tl-client.sh status` | Check daemon health |
| `./tl-client.sh info` | Storage stats |

### Required Workflow

1. **Start of session**: `./tl-client.sh setup`
2. **After successful changes**: `./tl-client.sh save "description"`
3. **Before risky work**: `./tl-client.sh save` (create restore point)
4. **If something breaks**: `./tl-client.sh log` then `./tl-client.sh restore <id>`
5. **Major milestones**: `./tl-client.sh pin <id> <milestone-name>`

### Key Facts

- Automatic checkpoints every 5 seconds when files change
- Restore is instant (<100ms)
- `save` is cheap - use liberally
- Content-addressed storage with deduplication
