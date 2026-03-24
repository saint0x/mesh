## Version Control: TL Client (Timelapse)

**USE `./tl-client.sh` FOR ALL VERSION CONTROL - NOT GIT**

TL replaces git completely. Automatic checkpoints + JJ integration for remote push/pull.

### Commands

| Command | Purpose |
|---------|---------|
| `setup` | Initialize + start daemon |
| `save` | Force immediate checkpoint |
| `log [n]` | View checkpoint history |
| `restore <id>` | Restore to checkpoint |
| `diff <a> <b>` | Compare checkpoints |
| `pin <id> <name>` | Name checkpoint |
| `unpin <name>` | Remove pin |
| `gc` | Garbage collection |
| `push` | Push to remote (via JJ) |
| `pull` | Pull from remote (via JJ) |
| `publish <id>` | Publish checkpoint for push |
| `worktree` | Manage workspaces |

### Workflow

1. `./tl-client.sh setup` - Once per repo
2. Make changes
3. `./tl-client.sh save` - Checkpoint
4. `./tl-client.sh publish HEAD` - Prepare for push
5. `./tl-client.sh push` - Push to remote

### Restore Workflow

```
./tl-client.sh log           # Find checkpoint
./tl-client.sh restore <id>  # Restore
```

### Key Facts

- Auto-checkpoints every 5 seconds
- Restore is instant (<100ms)
- Push/pull via JJ integration
- `save` is cheap - use liberally
