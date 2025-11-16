function setFadeVisibility(element, hidden) {
    if (hidden) {
        element.classList.remove("collapsible-visible");
        element.classList.add("collapsible-hidden");
        // After fade-out, fully hide from layout (CSS cannot do this cleanly)
        setTimeout(() => { element.style.display = "none"; }, 150);
    } else {
        element.style.display = "inline";
        element.classList.remove("collapsible-hidden");
        element.classList.add("collapsible-visible");
    }
}

function toggleChildren(name, hidden) {
    const node = document.getElementById("class-" + name);
    if (!node) return;

    setFadeVisibility(node, hidden);

    const edges = document.querySelectorAll(`.edge-group[data-source='${name}']`);
    edges.forEach(edge => {
        const target = edge.getAttribute("data-target");
        setFadeVisibility(edge, hidden);
        toggleChildren(target, hidden);
    });
}

function toggleNode(name) {
    const node = document.getElementById("class-" + name);
    if (!node) return;

    const marker = document.getElementById("toggle-" + name);

    const collapsed = node.getAttribute("data-collapsed") === "true";
    const newState = !collapsed;
    node.setAttribute("data-collapsed", newState ? "true" : "false");

    // Update triangle ▼▶
    if (marker) {
        marker.textContent = newState ? "\u25B6" : "\u25BC";
    }

    const edges = document.querySelectorAll(`.edge-group[data-source='${name}']`);
    edges.forEach(edge => {
        const target = edge.getAttribute("data-target");
        setFadeVisibility(edge, newState);
        toggleChildren(target, newState);
    });
}