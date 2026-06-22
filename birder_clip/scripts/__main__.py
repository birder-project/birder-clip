import pkgutil

import birder_clip.scripts


def list_scripts() -> list[str]:
    scripts = []
    for _, name, is_pkg in pkgutil.iter_modules(birder_clip.scripts.__path__):
        if name.startswith("_") is True:  # Skip private modules
            continue

        if is_pkg is False:
            scripts.append(name)

    return scripts


def main() -> None:
    print("Available birder scripts:")
    for script in list_scripts():
        print(f"    birder_clip.scripts.{script}")


if __name__ == "__main__":
    main()
