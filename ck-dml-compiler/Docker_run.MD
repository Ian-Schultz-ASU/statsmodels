# Running Docker container and and compute results 

```bash
# 1. Open Docker Desktop and wait until it says "Engine running".
#    In Settings > Resources, set Memory to at least 8 GB (the big run uses about 2 GB).
open -a Docker

# 2. Go to the repo and get the latest code
cd ~/Desktop/notes/classes/ml_systems598/code/ml-compiler-project
git pull

# 3. Build the image, compile dml-opt, and run all tests.
#    The first run downloads and builds the image, which takes several minutes.
#    Expect "8 passed, 0 failed" and "207 passed".
./dev make test

# 4. One-minute sanity run of the benchmark (nothing is saved). 
#   Reuses same image (unless removed), and it also reuses the dml-opt that make test compiled
#    Expect a last line starting with "done:".
./dev make bench-quick

# 5. The full chain: benchmark, EconML profile, figures, report (about 15 minutes).
#     reates new results, and in results/
#    Close other heavy apps and plug in power, so the timings are steady.
./dev make results

# 6. Look at what it made
open results/figures            # the four PNG figures
open docs/RESULTS.md            # the tables and limitations
git status                      # shows exactly which files changed
```

### Cleanup 

```bash
# Remove this project's stopped containers and its network (keeps the image and build volume)
docker compose -f docker/compose.yaml down

# Wipe only the compiled dml-opt, so the next "make build" starts fresh
./dev make clean

# Remove containers AND the build volume (dml-opt is rebuilt next time, about a minute)
docker compose -f docker/compose.yaml down --volumes

# Remove the image too (the next ./dev rebuilds it, several minutes)
docker image rm dmlc-dev

# Everything from this project in one go: containers, network, volume and image
docker compose -f docker/compose.yaml down --volumes --rmi all

# Free build cache left over from building the image (affects all your Docker projects)
docker builder prune
```

### Other

One thing to watch: compose does not rebuild the image on its own after you change requirements.txt or docker/Dockerfile.dev. After such a change, rebuild it: 

`docker compose -f docker/compose.yaml build`